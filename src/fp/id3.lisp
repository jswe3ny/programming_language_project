
(defstruct node
  col-idx is-leaf label children min-v max-v)

(defun calculate-entropy (y-ids)
  (let ((total (length y-ids)))
    (if (= total 0) 0.0
        (let ((counts (make-hash-table)))
          (dolist (y y-ids) (incf (gethash y counts 0)))
          (let ((entropy 0.0))
            (maphash #'(lambda (k count)
                         (let ((p (/ count (float total))))
                           (incf entropy (* (- p) (log p 2)))))
                     counts)
            entropy)))))

(defun most-common-label (y-ids)
  (if (null y-ids) 0
      (let ((c0 (count 0 y-ids)) (c1 (count 1 y-ids)))
        (if (>= c0 c1) 0 1))))

(defun get-col-bounds (data col-idx)
  (let ((vals (remove-if-not #'numberp (mapcar #'(lambda (r) (nth col-idx r)) data))))
    (if (null vals) (list 0 0) (list (apply #'min vals) (apply #'max vals)))))

(defun get-bin (val min-v max-v bins)
  (if (not (numberp val)) val
      (let* ((range (- max-v min-v))
             (step (if (> range 0) (/ range bins) 1.0))
             (bin (floor (/ (- val min-v) step))))
        (cond ((< bin 0) 0) ((>= bin bins) (1- bins)) (t bin)))))

(defun filter-data (data y-data col-idx val min-v max-v bins)
  (let ((sub-x nil) (sub-y nil))
    (loop :for r :in data :for y :in y-data :do
      (let ((v (nth col-idx r)))
        (when (equal (if (numberp v) (get-bin v min-v max-v bins) v) val)
          (push r sub-x) (push y sub-y))))
    (list (nreverse sub-x) (nreverse sub-y))))

(defun calc-gain (data y-data col-idx min-v max-v bins)
  (let* ((tot (length y-data))
         (root-h (calculate-entropy y-data))
         (vals (make-hash-table :test 'equal))
         (weighted-h 0.0))
    (dolist (r data)
      (let ((v (nth col-idx r)))
        (setf (gethash (if (numberp v) (get-bin v min-v max-v bins) v) vals) t)))
    (maphash #'(lambda (v k)
                 (let* ((subs (filter-data data y-data col-idx v min-v max-v bins))
                        (sy (second subs)))
                   (incf weighted-h (* (/ (length sy) (float tot)) (calculate-entropy sy)))))
             vals)
    (- root-h weighted-h)))

(defun id3-train (data y-data cols depth max-d bins)
  (cond 
    ((or (null y-data) (= (calculate-entropy y-data) 0.0) (>= depth max-d) (null cols))
     (make-node :is-leaf t :label (most-common-label y-data)))
    (t
     (let ((best-g -1) (best-c -1) (best-min 0) (best-max 0))
       (dolist (c cols)
         (let* ((b (get-col-bounds data c))
                (g (calc-gain data y-data c (first b) (second b) bins)))
           (when (> g best-g) 
             (setf best-g g best-c c best-min (first b) best-max (second b)))))
       
       (if (< best-g 1e-4) 
           (make-node :is-leaf t :label (most-common-label y-data))
           (let ((node (make-node :col-idx best-c :is-leaf nil :label (most-common-label y-data) :min-v best-min :max-v best-max :children (make-hash-table :test 'equal)))
                 (vals (make-hash-table :test 'equal)))
             (dolist (r data)
               (let ((v (nth best-c r)))
                 (setf (gethash (if (numberp v) (get-bin v best-min best-max bins) v) vals) t)))
             (maphash #'(lambda (v k)
                          (let* ((subs (filter-data data y-data best-c v best-min best-max bins)))
                            (setf (gethash v (node-children node))
                                  (id3-train (first subs) (second subs) (remove best-c cols) (1+ depth) max-d bins))))
                      vals)
             node))))))

(defun predict-id3-row (row node bins)
  (if (node-is-leaf node) (node-label node)
      (let* ((val (nth (node-col-idx node) row))
             (key (if (numberp val) (get-bin val (node-min-v node) (node-max-v node) bins) val))
             (child (gethash key (node-children node))))
        (if child (predict-id3-row row child bins) (node-label node)))))
