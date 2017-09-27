(ns learnclojure.core
  (:import [java.util Date Calendar])
  (:require [clojure.core.async :refer [chan go go-loop >! <!] :as async]
            [learnclojure.matrix :as matrix]))

;------------------------------------------------------------------------------------------
(defn matrix [array]
  "create a matrix from an array [1 2 3] => [[1] [2] [3]]"
  (mapv #(vector %) array))

;------------------------------------------------------------------------------------------
(defn activation-fn [x]
  "activation function"
  (Math/tanh x))

(defn dactivation-fn [y]
  "derivation of activation function"
  (- 1.0 (* y y)))

(defn layer-calculation [inputs weights]
  "forward propagate the input of a layer"
    (mapv #(mapv activation-fn %)
        (matrix/mmul weights inputs)))

(defn output-delta-calculation [outputs targets]
  "calculate the delta errors for the output layer (target value - actual value)"
  (mapv * (mapv dactivation-fn outputs) (mapv - targets outputs)))

(defn layer-delta-calculation [neurons deltas weights]
  (let [n (mapv dactivation-fn neurons)
        d (flatten (matrix/mmul deltas weights))]
    (mapv * n d)))

(defn update-weights-calculation [neurons deltas weights learning-rate]
  (let [n (matrix/mmul neurons deltas)]
    (println "n=" n)
    (println "w=" weights)
    (+ (flatten n) (map #(* learning-rate %) (flatten weights)))))

; (+ strengths (* lrate (mapv #(* deltas %) neurons))))

;------------------------------------------------------------------------------------------
(def input-neurons [1 0])

(def input-hidden-weights [[0.12 0.01]
                           [0.20 0.02]
                           [0.13 0.03]])

(def hidden-output-weights [[0.15 0.02 0.01]
                            [0.16 0.03 0.02]])

(def targets [0 1])

(def learning-rate 0.1)
;------------------------------------------------------------------------------------------

(def hidden-neurons (layer-calculation (matrix input-neurons) input-hidden-weights))
(def output-neurons (layer-calculation hidden-neurons hidden-output-weights))
(def output-deltas (output-delta-calculation (flatten output-neurons) targets))
(def hidden-deltas (layer-delta-calculation (flatten hidden-neurons) (vector output-deltas) hidden-output-weights))
(def new-hidden-output-weights (update-weights-calculation hidden-neurons (vector hidden-deltas) hidden-output-weights learning-rate))

output-neurons
;=> [[0.02315019005321053]
;    [0.027608061500083565]]

hidden-neurons
;=> [[0.11942729853438588]
;    [0.197375320224904]
;    [0.12927258360605834]]

output-deltas
;=> [-0.023137783141771645 0.9716507764442904]

hidden-deltas
;=> [0.14982559238071416 0.027569216735265096 0.018880751432503236]

new-hidden-output-weights

(def deltas (vector output-deltas))
(def weights hidden-output-weights)
(def neurons (flatten hidden-neurons))



