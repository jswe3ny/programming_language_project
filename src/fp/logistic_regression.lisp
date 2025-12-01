
(defun sigmoid (z)
  "The activation function: 1 / (1 + e^-z)"
  (if (> z 100) 1.0
      (if (< z -100) 0.0
          (/ 1.0 (+ 1 (exp (- z)))))))

(defun dot-product (v1 v2)
  "Calculates dot product of two lists."
  (loop :for x :in v1 
        :for w :in v2 
        :sum (* x w)))

(defun predict-proba (row weights bias)
  (let ((z (+ (dot-product row weights) bias)))
    (sigmoid z)))

(defun logistic-predict (row weights bias)
  (if (>= (predict-proba row weights bias) 0.5) 1 0))

(defun train-logistic (x-train y-train lr epochs l2)
    "Logistic Regression training" 
  (let* ((n-features (length (first x-train)))
         (m (float (length x-train)))
         (weights (make-list n-features :initial-element 0.0))
         (bias 0.0))
    
    (loop :for epoch :from 1 :to epochs :do
      (let ((dw (make-list n-features :initial-element 0.0))
            (db 0.0))
        
        (loop :for row :in x-train
              :for y :in y-train
              :do (let* ((pred (predict-proba row weights bias))
                         (error (- pred y)))
                    (incf db error)
                    (setf dw (mapcar #'(lambda (curr-dw x-val) 
                                         (+ curr-dw (* error x-val)))
                                     dw row))))
        
        (setf weights (mapcar #'(lambda (w d)
                                  (- w (* lr (+ (/ d m) (* l2 w)))))
                              weights dw))
        (setf bias (- bias (* lr (/ db m))))
        
))
            
    (list weights bias)))
