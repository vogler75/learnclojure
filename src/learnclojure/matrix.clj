(ns learnclojure.matrix)

(def transpose (fn [m]
  (apply map vector m)))

(def mmul (fn [a b]
  (let [nested-for (fn [f x y] (map (fn [a] (map (fn [b] (f a b)) y)) x))]
    (nested-for (fn [x y] (reduce + (map * x y))) a (transpose b)))))

(def ma [[1 2 3 4]
         [4 3 2 1]
         [4 3 2 1]])

(def mb [[1]
         [2]
         [3]
         [4]])

(mmul ma mb)
