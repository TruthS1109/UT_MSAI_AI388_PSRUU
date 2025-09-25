(define (domain cake)
  (:requirements :strips)
  (:predicates
    (have ?c)
    (eaten ?c)
    (dirty)
  )
  (:action eat
      :parameters (?c)
      :precondition (have ?c)
      :effect (and
          (not (have ?c))
          (eaten ?c)
      )
  )
  (:action bake
      :parameters (?c)
      :precondition (and (not (have ?c)) (not (dirty)))
      :effect (and (have ?c) (dirty))
  )
 (:action clean
      :parameters ()
      :precondition (dirty)
      :effect (not (dirty))
  )
)