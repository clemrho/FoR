(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects a l h b e c k f g)
(:init 
(handempty)
(ontable a)
(ontable l)
(ontable h)
(ontable b)
(ontable e)
(ontable c)
(ontable k)
(ontable f)
(ontable g)
(clear a)
(clear l)
(clear h)
(clear b)
(clear e)
(clear c)
(clear k)
(clear f)
(clear g)
)
(:goal
(and
(on a l)
(on l h)
(on h b)
(on b e)
(on e c)
(on c k)
(on k f)
(on f g)
)))