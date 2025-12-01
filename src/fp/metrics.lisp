
(defun calc-accuracy (predictions answers)
    (let ((correct 0.0)
          (totalPredictions (length predictions)))
      (loop :for pred :in predictions
            :for act :in answers
            :do (if (= pred act) (incf correct)))
      (if (> totalPredictions 0)
            (/ correct (float totalPredictions ))
            0.0)))

(defun calc-rmse (predictions answers)
  (let ((sum-sq-diff 0.0) (n (length predictions)))
    (loop :for p :in predictions :for a :in answers :do (incf sum-sq-diff (expt (- p a) 2)))
    (if (> n 0) (sqrt (/ sum-sq-diff n)) 0.0)))

(defun calc-r2 (predictions answers)
  (let* ((n (length answers))
         (mean-y (if (> n 0) (/ (loop :for a :in answers :sum a) (float n)) 0.0))
         (ss-tot (loop :for a :in answers :sum (expt (- a mean-y) 2)))
         (ss-res (loop :for p :in predictions :for a :in answers :sum (expt (- a p) 2))))
    (if (> ss-tot 0) (- 1 (/ ss-res ss-tot)) 0.0)))

(defun calc-macro-f1 (predictions answers)
  "Calculates Macro-F1 score (Average F1 of Class 0 and Class 1)."
  (let ((tp 0) (fp 0) (fn 0) (tn 0))
    
    ;; 1. Count True Positives, False Positives, False Negatives, True Negatives
    (loop :for pred :in predictions
          :for act :in answers
          :do (cond
                ((and (= pred 1) (= act 1)) (incf tp))
                ((and (= pred 1) (= act 0)) (incf fp))
                ((and (= pred 0) (= act 1)) (incf fn))
                ((and (= pred 0) (= act 0)) (incf tn))))
    
    (let* ((prec1 (if (> (+ tp fp) 0) (/ tp (float (+ tp fp))) 0.0))
           (rec1  (if (> (+ tp fn) 0) (/ tp (float (+ tp fn))) 0.0))
           (f1-1  (if (> (+ prec1 rec1) 0) (* 2 (/ (* prec1 rec1) (+ prec1 rec1))) 0.0))
           
           ;; calcs  F1 for Class 0
           (prec0 (if (> (+ tn fn) 0) (/ tn (float (+ tn fn))) 0.0))
           (rec0  (if (> (+ tn fp) 0) (/ tn (float (+ tn fp))) 0.0))
           (f1-0  (if (> (+ prec0 rec0) 0) (* 2 (/ (* prec0 rec0) (+ prec0 rec0))) 0.0)))
      
      ;;  Macro F1 is the average of both
      (/ (+ f1-1 f1-0) 2.0))))

(defun get-sloc (filename)
  "Counts valid lines of code in a specific file. Returns 0 if file is missing."
  (let ((count 0))
    (if (probe-file filename)
        (with-open-file (stream filename :direction :input)
          (loop :for line = (read-line stream nil)
                :while line
                :do (let ((clean (string-trim '(#\Space #\Tab #\Newline #\Return) line)))
                      ;; Check if line is NOT empty AND NOT a comment (;)
                      (when (and (> (length clean) 0)
                                 (not (char= (char clean 0) #\;)))
                        (incf count)))))
        ;; If file doesn't exist, return 0 (or print warning if you prefer)
        0)
    count))