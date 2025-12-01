

(defun euclidean-distance (row1 row2)
  (sqrt
    (loop :for x :in row1
          :for y :in row2
          :for diff = (- x y)
          :sum (* diff diff)
          )
    )
)

(defun knn (x-train y-train test-row k)
  "Predicts the class (0 or 1) for a single test-row."  
  ;; gets distances from test-row to other training rows
  (let* ((distances (mapcar #'(lambda (train-row)
                                (euclidean-distance test-row train-row))
                            x-train))
         
         ;;pair distances with their actual labels (y-train)
         (dist-label-pairs (mapcar #'list distances y-train))
         
         ;; Sorts the pairs by distance (smallest to largest)
         (sorted-pairs (sort dist-label-pairs #'< :key #'first))
         
         ;; retunrs top k neighbors
         (k-nearest (subseq sorted-pairs 0 k))
         
         ;;  extract labels
         (k-labels (mapcar #'second k-nearest))
         
         ;; counts the votes
         (votes-for-0 (count 0 k-labels))
         (votes-for-1 (count 1 k-labels)))
    
    (if (>= votes-for-0 votes-for-1) 0 1)))

