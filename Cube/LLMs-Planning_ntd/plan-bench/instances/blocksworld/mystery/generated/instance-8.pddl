(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects j e h d f i a)
(:init 
(harmony)
(planet j)
(planet e)
(planet h)
(planet d)
(planet f)
(planet i)
(planet a)
(province j)
(province e)
(province h)
(province d)
(province f)
(province i)
(province a)
)
(:goal
(and
(craves j e)
(craves e h)
(craves h d)
(craves d f)
(craves f i)
(craves i a)
)))