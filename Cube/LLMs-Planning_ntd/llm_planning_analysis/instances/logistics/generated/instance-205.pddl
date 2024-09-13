(define (problem LG-generalization)
(:domain logistics-strips)(:objects c0 t0 a0 l0-4 p0 l0-3 p2 l0-6 p3 l0-2 p4 l0-1 p1 l0-5 p6 l0-0 p5)
(:init 
(CITY c0)
(TRUCK t0)
(AIRPLANE a0)
(LOCATION l0-4)
(in-city l0-4 c0)
(OBJ p0)
(at p0 l0-4)
(at t0 l0-4)
(LOCATION l0-3)
(in-city l0-3 c0)
(OBJ p2)
(at p2 l0-3)
(LOCATION l0-6)
(in-city l0-6 c0)
(OBJ p3)
(at p3 l0-6)
(LOCATION l0-2)
(in-city l0-2 c0)
(OBJ p4)
(at p4 l0-2)
(LOCATION l0-1)
(in-city l0-1 c0)
(OBJ p1)
(at p1 l0-1)
(LOCATION l0-5)
(in-city l0-5 c0)
(OBJ p6)
(at p6 l0-5)
(LOCATION l0-0)
(in-city l0-0 c0)
(OBJ p5)
(at p5 l0-0)
(AIRPORT l0-0)
(at a0 l0-0)
)
(:goal
(and
(at p0 l0-3)
(at p2 l0-6)
(at p3 l0-2)
(at p4 l0-1)
(at p1 l0-5)
(at p6 l0-0)
)))