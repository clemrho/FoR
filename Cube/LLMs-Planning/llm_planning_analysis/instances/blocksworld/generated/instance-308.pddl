(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects l k j a b g f c)
(:init 
(handempty)
(ontable l)
(ontable k)
(ontable j)
(ontable a)
(ontable b)
(ontable g)
(ontable f)
(ontable c)
(clear l)
(clear k)
(clear j)
(clear a)
(clear b)
(clear g)
(clear f)
(clear c)
)
(:goal
(and
(on l k)
(on k j)
(on j a)
(on a b)
(on b g)
(on g f)
(on f c)
)))