;---------------------------------------------------------------------------------
; Atoms

(def visitors (atom (list ())))

(defn hello [username]
  (swap! visitors conj username)
  (str "Hello, " username))

;---------------------------------------------------------------------------------
; Matrix Multiplication

; (defn transpose [s]
;  (apply map vector s))

;(defn nested-for [f x y]
;  (map (fn [a] (map (fn [b] (f a b)) y)) x))

(def mmul (fn [a b]
  (let [nested-for (fn [f x y] (map (fn [a] (map (fn [b] (f a b)) y)) x))
        transpose (fn [m] (apply map vector m))]
    (nested-for (fn [x y] (reduce + (map * x y))) a (transpose b)))))

(def ma [[1 2 3 4]
         [4 3 2 1]])

(def mb [[1]
         [2]
         [3]
         [4]])

;---------------------------------------------------------------------------------
; Arguments

(def fu1 (fn [{name :name}] (str name)))

(def fu2 (fn [[name age]] (str "hello" name " of age " age)))

(defn fu3 [& {:keys [foo bar] :or {foo "foo-default" bar "bar-default"}}]
  {:output-foo foo :output-bar bar})

;---------------------------------------------------------------------------------
; Reduce, Filter, ...

(defn filter-even [acc val]
  (if (even? val) (conj acc val) acc))

(reduce filter-even [] [0 1 2 3 4 5])

(filter even? [0 1 2 3 4 5])

(map (fn [val] (+ val 2)) [0 1 2 3 4 5])

(defn group-even [acc val]
  (let [key (if (even? val) :even :odd)]
    (update-in acc [key] (fn [old] (conj old val)))))

(reduce group-even {} [0 1 2 3 4 5 6])

(group-by (fn [val] (if (even? val) :even :odd)) [0 1 2 3 4 5])

;---------------------------------------------------------------------------------
; Java Interop

(def d1 (doto (Calendar/getInstance)
          (.set Calendar/YEAR 1985)
          (.set Calendar/MONTH 9)
          (.set Calendar/DATE 26)))

(set! *warn-on-reflection* true)

(defn strlen [^java.lang.String s] (.length s))
(defn badstrlen [s] (.length s))

;(time (reduce + (map badstrlen (repeat 100000 "abc"))))
;(time (reduce + (map strlen (repeat 100000 "abc"))))

; Thread mythread = new Thread () { public void run() { println... } }
(def mythread (proxy [Thread] [] (run [] (println "Running in a thread"))))
;(.run mythread)

(def mythread2 (reify Runnable (run [this] (println "Running reify"))))
;(.run mythread2)

;(.start (Thread. (fn [] (println "Hello Thread"))))

;---------------------------------------------------------------------------------
(defn debug [s] (str s " " (Date.)))

(defn fu [nr] (do (print (str nr)) (Thread/sleep (* nr 1000)) (debug nr)))

(map deref [(future (fu 1))
            (future (fu 2))
            (future (fu 3))])

(defn myfn [] (delay (fu 1))) ; "lazy" function, computed when deref is called

(deref (myfn))

;---------------------------------------------------------------------------------
(defn fib [^long n] (if (< n 2) 1 (+ (fib (- n 1)) (fib (- n 2)))))

(def fibm (memoize fib))


;(str "fib-time " (time (fibm 40)))

;---------------------------------------------------------------------------------
; Main
  (defn -main [& args]
    ;(println (mmul ma mb))
    ;(println (fu2 ["vogler" 25]))
    ;(println (fu3 :foo "hello"))
    ;(println d1)
    (time (fibm 40))
    (time (fibm 40)))
