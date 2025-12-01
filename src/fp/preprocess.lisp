
(defun split-by-comma (string)
  "Splits STRING into a list of strings, separated by a comma."
  (loop :for start = 0 :then (1+ end)
        :for end = (position #\, string :start start) ;splits by comma - for CSV
        :collect (subseq string start end)
        :if (null end)
            :do(loop-finish)))


(defun load-csv (filepath)
  "Loads a CSV from FILEPATH, skips the header, and *returns* a list of rows."
  (let ((data nil))
    (with-open-file (stream filepath :direction :input)
      (read-line stream nil)
      (loop :for line = (read-line stream nil)
            :while line
            :do (push (split-by-comma line) data)))
    (nreverse data)))


(defun convert-row-to-mixed-types (string-row)
  "Takes a single row (list of strings) and converts the
   known numeric columns into actual numbers."
  
  ;; 0=age, 3=education.num, 9=capital.gain,
  ;; 10=capital.loss, 11=hours.per.week
  (let ((numeric-indices '(0 3 9 10 11)))
    
    (loop :for i :from 0            
          :for item :in string-row 
          
          :collect (if (member i numeric-indices)
                       (parse-integer item :junk-allowed t)
                       
                       ;; Otherwise, just keep the original string
                       item))))




(defun build-ohe-maps (mixed-data)

  ;; From th CSV, categorical indices are:
  ;; 1(workclass), 2(education), 4(marital.status), 5(occupation),
  ;; 6(relationship), 7(race), 8(sex), 12(native.country)
  (let ((categorical-indices '(1 2 4 5 6 7 8 12))
        (ohe-maps (make-hash-table)))

    ;; Find all unique values
    (dolist (row mixed-data)
      (dolist (col-index categorical-indices)
        (let* ((value (nth col-index row))
               (map (gethash col-index ohe-maps (make-hash-table :test 'equal))))
          (setf (gethash value map) t)
          (setf (gethash col-index ohe-maps) map))))

    ;; Convert unique values from a map to a sorted list
    (loop :for col-index :in categorical-indices
          :do (let* ((map (gethash col-index ohe-maps))
                     (keys nil))
                (maphash (lambda (key val) (push key keys)) map)
                (setf (gethash col-index ohe-maps) (sort keys #'string<))))
    
    ohe-maps))


(defun apply-ohe (row ohe-maps)
  "takes one row and the 'ohe-maps',
   and returns a new, fully-numeric row."

  (let ((categorical-indices '(1 2 4 5 6 7 8 12))
        (target-index 13))

    (loop :for item :in row
          :for i :from 0

          ;; --- Case 1: Is this the 'income' column? ---
          :if (= i target-index)
            :do (progn)

          :else :if (member i categorical-indices)
            :append (let* ((categories (gethash i ohe-maps))
                           (pos (position item categories :test #'string=)))
                      ;; This inner loop creates and returns the (0 0 1 0) list
                      (loop :for j :from 0 :below (length categories)
                            :collect (if (= j pos) 1 0)))

          :else
            :collect item)))


(defun create-index-list (n)
  "Creates a list of numbers from 0 to n-1."
  (loop :for i :from 0 :below n :collect i))

(defun shuffle-list (list rng)
  "Returns a shuffled list"
  ;; 'copy-list' is crucial for immutability.
  (let ((shuffled (copy-list list))
        (n (length list)))

    (loop :for i :from (1- n) :downto 1
          ;; This 'random' call now uses the 'rng' we passed in
          :do (let ((j (random (1+ i) rng)))
                (rotatef (nth i shuffled) (nth j shuffled))))

    shuffled))


(defun build-subset (data indices)
  "Builds a new list by selecting items from 'data'
   based on the 'indices' list."

  (loop :for i :in indices
        :collect (nth i data)))

(defun trim-string (s)
  (string-trim '(#\Space #\Tab #\Newline #\Return #\. #\") s))

(defun process-target-column (row)
 (let* ((raw-val (nth 13 row))
         (val-str (format nil "~a" raw-val))
         ;; WE ADDED THIS LINE:
         (clean-val (trim-string val-str))) 
    
    ;; Now this comparison actually works
    (if (string= clean-val "<=50K") 0 1)))

(defun calculate-mean (list-of-numbers)
  "Calculates the mean of a list of numbers."
  (if (null list-of-numbers)
      0 ; Handle empty list
      (/ (loop :for x :in list-of-numbers :sum x)
         (float (length list-of-numbers)))))

(defun transpose (list-of-lists)
  "Flips a list of rows into a list of columns."
  ;; 'apply' with 'mapcar' is the classic Lisp transpose.
  (apply #'mapcar #'list list-of-lists))

(defun calculate-std-dev (list-of-numbers mean)
  "Calculates the population standard deviation."
  (let* ((n (length list-of-numbers))
         ;; 1. Get sum of squared differences from mean
         (sum-of-squares (loop :for x :in list-of-numbers
                               :sum (expt (- x mean) 2)))
         ;; 2. Get variance
         (variance (if (> n 0) (/ sum-of-squares (float n)) 0))
         ;; 3. Get std dev
         (std (sqrt variance)))
    
    ;; This handles the Python notebook's 'eps' rule
    ;; to prevent dividing by zero if a column has no variance.
    (if (< std 1e-8) 1.0 std)))

(defun calculate-column-stats (x-data)
  "Calculates the mean and std-dev
   for *every column* in a dataset."
  
  (let ((data-by-cols (transpose x-data)))
    
    ;; 2. Loop over each *column*
    (mapcar #'(lambda (column)
                ;; 3. Calculate its stats
                (let ((mean (calculate-mean column)))
                  ;; 4. Return a new list (mean std-dev)
                  (list mean (calculate-std-dev column mean))))
            data-by-cols)))



(defun apply-normalization-to-row (row norm-stats)
  "Applies the z-score formula to one row
   using the learned 'norm-stats'."

  ;; This loop walks down the 'row' and 'norm-stats'
  ;; lists at the same time.
  (loop
    :for x :in row             ;<-- 'x' is the value (e.g., 26)
    :for stats :in norm-stats  ;<-- 'stats' is the rule (e.g., (38.68 13.21))

    :collect (let ((mean (first stats))
                   (std (second stats)))

               ;; Apply the z-score formula: (x - mean) / std
               (/ (- x mean) std))))


