(ns learnclojure.core
  (:import [java.util Date Calendar])
  (:require [clojure.core.async :refer [chan go go-loop >! <!] :as async]
            [learnclojure.matrix :as matrix]))

(defn activation-fn [x]
  "activation function"
  (Math/tanh x))

(defn dactivation-fn [y]
  "derivation of activation function"
  (- 1.0 (* y y)))

(defn layer-activation [inputs weights]
  "forward propagate the input of a layer"
    (mapv #(mapv activation-fn %)
        (matrix/mmul weights inputs)))

(defn output-deltas [outputs targets]
  "calculate the delta errors for the output layer (desired value - actual value)"
  (mapv * (mapv dactivation-fn outputs) (mapv - targets outputs)))

;------------------------------------------------------------------------------------------
(defn matrix-of [array]
  "create a matrix from an array [1 2 3] => [[1] [2] [3]]"
  (mapv #(vector %) array))

;------------------------------------------------------------------------------------------
(def input-neurons [0.1, 0.2])

(def input-hidden-weights [[1 2]
                           [3 4]
                           [5 6]])

(def hidden-output-weights [[0.15 0.16 0.17]
                            [0.02 0.03 0.04]])

(def targets [0.2, 0.3])

(def hidden-neurons (layer-activation (matrix-of input-neurons) input-hidden-weights))
(def output-neurons (layer-activation hidden-neurons hidden-output-weights))

(output-deltas (flatten output-neurons) targets)



