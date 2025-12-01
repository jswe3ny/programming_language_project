
(defun gaussian-log-pdf (x mean var)
  "Vectorized log-density of a diagonal Gaussian.
   Matches Python: _gaussian_logpdf"
  (let* ((eps 1e-9)
         ;; Safety: Ensure variance is never zero
         (safe-var (max var eps))
         ;; Term 1: -0.5 * log(2 * pi * var)
         (term1 (* -0.5 (log (* 2 pi safe-var))))
         ;; Term 2: -0.5 * ((x - mean)^2 / var)
         (term2 (* -0.5 (/ (expt (- x mean) 2) safe-var))))
    
    (+ term1 term2)))

(defun separate-by-class (x-data y-data)
  "Splits data into Class 0 and Class 1 lists."
  (let ((class0 nil) (class1 nil))
    (loop :for row :in x-data
          :for label :in y-data
          :do (if (= label 0)
                  (push row class0)
                  (push row class1)))
    (list class0 class1)))

(defun train-nb (x-train y-train)
  "Trains Naive Bayes. Returns (Priors Means Vars)."
  
  (let* ((separated (separate-by-class x-train y-train))
         (c0 (first separated))
         (c1 (second separated))
         (total (float (length x-train)))
         
        
         ;; causes 'infinite penalty' errors. 1e-2 is standard for sparse data.
        ;; (var-smoothing 1e-2)
         (var-smoothing 0.3)
         (prior0 (/ (length c0) total))
         (prior1 (/ (length c1) total))
         
         ;; Calculate Means and Variances for Class 0
         (stats0 (mapcar #'(lambda (stat) 
                             (let ((mean (first stat))
                                   (std (second stat)))
                               ;; Variance = std^2 + smoothing
                               (list mean (+ (* std std) var-smoothing)))) 
                         (calculate-column-stats c0)))
         
         ;; Calculate Means and Variances for Class 1
         (stats1 (mapcar #'(lambda (stat) 
                             (let ((mean (first stat))
                                   (std (second stat)))
                               (list mean (+ (* std std) var-smoothing)))) 
                         (calculate-column-stats c1))))
    
    (list prior0 prior1 stats0 stats1)))

(defun predict-nb-row (row model)
  "Predicts class 0 or 1 using Log-Likelihood."
  (let* ((prior0 (first model))
         (prior1 (second model))
         (stats0 (third model))
         (stats1 (fourth model))
         
         ;; Log Priors
         (log-prob0 (log (max prior0 1e-12)))
         (log-prob1 (log (max prior1 1e-12))))
    
    ;; Sum Likelihoods for Class 0
    (loop :for x :in row :for stat :in stats0
          :do (let ((mean (first stat))
                    (var (second stat)))
                (incf log-prob0 (gaussian-log-pdf x mean var))))

    ;; Sum Likelihoods for Class 1
    (loop :for x :in row :for stat :in stats1
          :do (let ((mean (first stat))
                    (var (second stat)))
                (incf log-prob1 (gaussian-log-pdf x mean var))))
    
    ;; Compare
    (if (> log-prob0 log-prob1) 0 1)))