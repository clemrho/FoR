(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects e b d g k a j c l f)
(:init 
(harmony)
(planet e)
(planet b)
(planet d)
(planet g)
(planet k)
(planet a)
(planet j)
(planet c)
(planet l)
(planet f)
(province e)
(province b)
(province d)
(province g)
(province k)
(province a)
(province j)
(province c)
(province l)
(province f)
)
(:goal
(and
(craves e b)
(craves b d)
(craves d g)
(craves g k)
(craves k a)
(craves a j)
(craves j c)
(craves c l)
(craves l f)
)))