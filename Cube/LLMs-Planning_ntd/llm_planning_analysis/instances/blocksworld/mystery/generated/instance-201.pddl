(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects h c a e j b g i f k l)
(:init 
(harmony)
(planet h)
(planet c)
(planet a)
(planet e)
(planet j)
(planet b)
(planet g)
(planet i)
(planet f)
(planet k)
(planet l)
(province h)
(province c)
(province a)
(province e)
(province j)
(province b)
(province g)
(province i)
(province f)
(province k)
(province l)
)
(:goal
(and
(craves h c)
(craves c a)
(craves a e)
(craves e j)
(craves j b)
(craves b g)
(craves g i)
(craves i f)
(craves f k)
(craves k l)
)))