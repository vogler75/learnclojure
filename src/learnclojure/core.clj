(ns learnclojure.core
  (:import [java.util Date Calendar])
  (:require [clojure.core.async :refer [chan go go-loop >! <!] :as async]
            [learnclojure.matrix :as matrix]))

;------------------------------------------------------------------------------------------
(defn matrix [array]
  "create a matrix from an array [1 2 3] => [[1] [2] [3]]"
  (mapv #(vector %) array))

;------------------------------------------------------------------------------------------
(defn activation-fn-tanh [x] "tanh activation" (Math/tanh x))

(defn dactivation-fn-tanh [y] "tanh derivative " (- 1.0 (* y y)))

(defn activation-fn-sigmoid [x] "sigmoid activation" (/ 1 (+ 1 (Math/exp (* -1 x)))))

(defn dactivation-fn-sigmoid [x] "sigmoid derivative" (/ (Math/exp x) (Math/pow (+ 1 (Math/exp x)) 2)))

(defn activation-fn [x] (activation-fn-tanh x))
(defn dactivation-fn [x] (dactivation-fn-tanh x))

;------------------------------------------------------------------------------------------
(defn layer-calculation [inputs weights]
  "forward propagate the input of a layer"
    (mapv #(mapv activation-fn %)
        (matrix/mmul weights inputs)))

(defn output-errors-calculation [outputs targets]
  "calculate the delta errors for the output layer (target value - actual value)"
  (mapv * (mapv dactivation-fn outputs) (mapv - targets outputs)))

(defn hidden-errors-calculation [errors weights]
  "calculate errors for the hidden layer based on the output-errors and weights"
  (flatten (matrix/mmul (matrix/transpose weights) (matrix errors))))

(defn update-weights-calculation [neurons weights errors rate]
  "update/adjust the weights of a hidden layer according to the errors"
  (:tbd))

;------------------------------------------------------------------------------------------
; Example 1
(def input-neurons [1 0])

(def input-hidden-weights [[0.12 0.01]
                           [0.20 0.02]
                           [0.13 0.03]])

(def hidden-output-weights [[0.15 0.02 0.01]
                            [0.16 0.03 0.02]])

(def targets [0 1])

(def learning-rate 0.2)

;------------------------------------------------------------------------------------------
; Example 2
(def input-neurons [0.90 0.10 0.80])

(def input-hidden-weights [[0.9 0.3 0.4]
                           [0.2 0.8 0.2]
                           [0.1 0.5 0.6]])

(def hidden-output-weights [[0.3 0.7 0.5]
                            [0.6 0.5 0.2]
                            [0.8 0.1 0.9]])

(def targets [0.50 0.20 0.10])

(def learning-rate 0.1)

;------------------------------------------------------------------------------------------
; Example 3
(def input-neurons [0.90 0.10])

(def input-hidden-weights [[0.1 0.2]
                           [0.3 0.4]])

(def hidden-output-weights [[2 3]
                            [1 4]])

(def targets [0.50 0.20])

(def learning-rate 0.1)

;------------------------------------------------------------------------------------------
(def hidden-neurons (layer-calculation (matrix input-neurons) input-hidden-weights))
(def output-neurons (layer-calculation hidden-neurons hidden-output-weights))

(def output-errors (output-errors-calculation (flatten output-neurons) targets))
(def hidden-errors (hidden-errors-calculation output-errors hidden-output-weights))

hidden-neurons
;=> [[0.11942729853438588]
;    [0.197375320224904]
;    [0.12927258360605834]]

output-neurons
;=> [[0.02315019005321053]
;    [0.027608061500083565]]

(def output-errors [0.8 0.5])
output-errors
;=> [-0.023137783141771645 0.9716507764442904]

hidden-errors
;=> [0.14982559238071416 0.027569216735265096 0.018880751432503236]


(def neurons hidden-neurons)
(def weights hidden-output-weights)
(def errors output-errors)
(def rate 0.2)

(def neurons [[0.4] [0.5]])
neurons
errors
weights

(mapv #(* errors %) (flatten neurons))
