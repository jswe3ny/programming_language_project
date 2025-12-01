


(defun mat-dot (v1 v2)
  "Dot product of two vectors."
  (loop :for x :in v1 :for y :in v2 :sum (* x y)))

(defun mat-transpose (m)
  "Transposes a matrix (flips rows and columns)."
  (apply #'mapcar #'list m))

(defun mat-mul (A B)
  "Multiplies Matrix A x Matrix B."
  (let ((Bt (mat-transpose B)))
    (mapcar #'(lambda (row)
                (mapcar #'(lambda (col) (mat-dot row col)) Bt))
            A)))

(defun mat-vec-mul (A v)
  "Multiplies Matrix A and  Vector v."
  (mapcar #'(lambda (row) (mat-dot row v)) A))

(defun add-bias-column (X)
  "Adds a column of 1.0s to the end of matrix X"
  (mapcar #'(lambda (row) (append row '(1.0))) X))

(defun add-ridge-penalty (A l2)
  "Adds L2 value to the diagonal of Matrix A (Ridge Regression)."
  (loop :for row :in A
        :for i :from 0
        :collect (loop :for val :in row
                       :for j :from 0
                       :collect (if (= i j) (+ val l2) val))))

;; --- 2. GAUSSIAN ELIMINATION SOLVER ---
(defun solve-linear-system (A b)
  (let* ((n (length A))
         ;; combines matrix A and b into one matrix
         (aug (loop :for i :below n 
                    :collect (append (nth i A) (list (nth i b))))))
    
    ;; forward elimination
    (loop :for i :below n :do
      (let ((pivot (nth i (nth i aug))))
        
       ;; (if (< (abs pivot) 1e-9) (setf pivot 1e-9))
        
        (loop :for j :from (1+ i) :below n :do
          (let ((factor (/ (nth i (nth j aug)) pivot)))
            (setf (nth j aug)
                  (loop :for k :below (1+ n)
                        :collect (- (nth k (nth j aug)) 
                                    (* factor (nth k (nth i aug))))))))))
    
    ;; back substitution
    (let ((x (make-list n :initial-element 0.0)))
      (loop :for i :from (1- n) :downto 0 :do
        (let ((sum 0.0))
          (loop :for j :from (1+ i) :below n :do
            (incf sum (* (nth j x) (nth j (nth i aug)))))
          (setf (nth i x) (/ (- (nth n (nth i aug)) sum) 
                             (nth i (nth i aug))))))
      x)))


(defun train-linear (x-train y-train l2)
  "Trains Linear Regression using the Normal Equation: w = (XtX + lI)^-1 Xt y"
 ;; (format t "Training Linear Regression (Closed Form)...~%")
  
  ;; appends bias col to X
  (let* ((X (add-bias-column x-train))
          ;; transposes X
         (Xt (mat-transpose X))
         ;; multiplies Xt and  X
         (XtX (mat-mul Xt X))
         ;; adds regularization
         (XtX-reg (add-ridge-penalty XtX l2))
         ;; multiplites  Xt and  y
         (Xty (mat-vec-mul Xt y-train)))
    
    (solve-linear-system XtX-reg Xty)))

(defun predict-linear (row weights)
  "Predicts a value. Row must NOT have the bias term yet."
  (mat-dot (append row '(1.0)) weights))
