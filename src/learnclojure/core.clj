(ns learnclojure.core
  (:import [java.util Date Calendar])
  (:require [clojure.core.async :refer [chan go go-loop >! <!] :as async]
            [learnclojure.matrix :as matrix]))

(def input-neurons [[1]
                    [2]])

(def input-hidden-strengths [[1 2]
                             [3 4]
                             [5 6]])

(def hidden-neurons [[0.0]
                     [0.0]
                     [0.0]])

(def hidden-output-strengths [[0.15 0.16]
                              [0.02 0.03]
                              [0.01 0.02]])

;(def activation-fn (fn [x] (Math/tanh x)))
(def activation-fn (fn [x] x))

(def dactivation-fn (fn [y] (- 1.0 (* y y))))

(defn layer-activation [inputs strengths]
  "forward propagate the input of a layer"
  (mapv activation-fn
    (mapv #(reduce + %)
        (matrix/mmul strengths inputs))))

(defn layer-activation [inputs strengths]
  "forward propagate the input of a layer"
    (mapv #(activation-fn %)
        (matrix/mmul strengths inputs)))


(matrix/mmul input-hidden-strengths input-neurons)

(layer-activation input-neurons input-hidden-strengths)
