(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l b f h i j e a k d)
(:init 
(harmony)
(planet l)
(planet b)
(planet f)
(planet h)
(planet i)
(planet j)
(planet e)
(planet a)
(planet k)
(planet d)
(province l)
(province b)
(province f)
(province h)
(province i)
(province j)
(province e)
(province a)
(province k)
(province d)
)
(:goal
(and
(craves l b)
(craves b f)
(craves f h)
(craves h i)
(craves i j)
(craves j e)
(craves e a)
(craves a k)
(craves k d)
)))