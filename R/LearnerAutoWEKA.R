#' @title Auto-WEKA Learner
#'
#' @description
#' Abstract base class for Auto-WEKA like learner.
#'
#' @param id (`character(1)`)\cr
#'   Identifier for the new instance.
#' @param task_type (`character(1)`)\cr
#'   Type of task, e.g. `"regr"` or `"classif"`.
#'   Must be an element of [mlr_reflections$task_types$type][mlr_reflections].
#' @param learner_ids (`character()`)\cr
#'  List of learner ids.
#' @param tuning_space (list of [paradox::TuneToken])\cr
#'  List of [paradox::TuneToken]s.
#' @param resampling ([mlr3::Resampling]).
#' @param measure ([mlr3::Measure]).
#' @param terminator ([bbotk::Terminator]).
#' @param callbacks (list of [mlr3tuning::CallbackTuning]).
#'
#' @export
LearnerAutoWEKA = R6Class("LearnerAutoWEKA",
  inherit = Learner,
  public = list(

    #' @field resampling ([mlr3::Resampling]).
    resampling = NULL,

    #' @field measure ([mlr3::Measure]).
    measure = NULL,

    #' @field terminator ([bbotk::Terminator]).
    terminator = NULL,

    #' @field callbacks (list of [mlr3tuning::CallbackTuning]).
    callbacks = NULL,

    #' @field learner_ids (`character()`).
    learner_ids = NULL,

    #' @field tuning_space (list of [TuneToken]).
    tuning_space = NULL,

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, task_type, learner_ids, tuning_space, resampling, measure, terminator, callbacks = list()) {
      self$learner_ids = assert_character(learner_ids)
      self$tuning_space = assert_list(tuning_space, types = "TuneToken")
      self$resampling = assert_resampling(resampling)
      self$measure = assert_measure(measure)
      self$terminator = assert_terminator(terminator)
      self$callbacks = assert_list(as_callbacks(callbacks), types = "CallbackTuning")
      assert_choice(task_type, mlr_reflections$task_types$type)

      # find packages
      learners = lrns(paste0(task_type, ".", self$learner_ids))
      learner_packages = unlist(map(learners, "packages"))
      packages = unique(c("mlr3tuning", "mlr3learners", "mlr3pipelines", "mlr3mbo", "mlr3autoweka", learner_packages))

      super$initialize(
        id = id,
        task_type = task_type,
        packages = packages,
        feature_types = mlr_reflections$task_feature_types,
        predict_types = names(mlr_reflections$learner_predict_types[[task_type]]),
        properties = mlr_reflections$learner_properties[[task_type]],
      )
    }
  ),

  private = list(

    .train = function(task) {

      # initialize graph learner
      gr_branch = get_branch_pipeline(self$task_type, self$learner_ids)
      graph = ppl("robustify", task = task, factors_to_numeric = TRUE) %>>% gr_branch
      graph_learner = as_learner(graph)
      graph_learner$id = "graph_learner"
      graph_learner$predict_type = self$measure$predict_type
      graph_learner$fallback = switch(self$task_type,
        "classif" = lrn("classif.featureless", predict_type = self$measure$predict_type),
        "regr" = lrn("regr.featureless"))

      # initialize search space
      search_space = get_search_space(self$task_type, self$learner_ids, self$tuning_space)

      # initialize mbo tuner
      surrogate = default_surrogate(n_learner = 1, search_space = search_space, noisy = TRUE)
      acq_function = AcqFunctionEI$new()
      acq_optimizer = AcqOptimizer$new(
        optimizer = opt("random_search", batch_size = 1000L),
        terminator = trm("evals", n_evals = 10000L))

      tuner = tnr("mbo",
        loop_function = bayesopt_ego,
        surrogate = surrogate,
        acq_function = acq_function,
        acq_optimizer = acq_optimizer,
        args = list(init_design_size = 10))

      # initialize auto tuner
      auto_tuner = auto_tuner(
        tuner = tuner,
        learner = graph_learner,
        resampling = self$resampling,
        measure = self$measure,
        terminator = self$terminator,
        search_space = search_space,
        evaluate_default = TRUE
      )

      auto_tuner$train(task)
      auto_tuner
    },

    .predict = function(task) {
      self$model$predict(task)
    }
  )
)

tuning_space_common_autoweka = list(
  # Decision Table
  decision_table.E     = to_tune(),
  decision_table.I     = to_tune(),
  decision_table.S     = to_tune(levels = c("BestFirst", "GreedyStepwise")),
  decision_table.X     = to_tune(1, 4),

  # KStar
  kstar.B     = to_tune(1, 100),
  kstar.E     = to_tune(),
  kstar.M     = to_tune(levels = c("a", "d", "m", "n")),

  # SGD
  sgd.F   = to_tune(),
  sgd.L   = to_tune(0.00001, 0.1, logscale = TRUE),
  sgd.R   = to_tune(1e-12, 10, logscale = TRUE),
  sgd.N   = to_tune(),
  sgd.M   = to_tune(),

  # MultilayerPerceptron
  multilayer_perceptron.L   = to_tune(0.1, 1),
  multilayer_perceptron.M   = to_tune(0.1, 1),
  multilayer_perceptron.B   = to_tune(),
  multilayer_perceptron.H   = to_tune(levels = c("a", "i", "o", "t")),
  multilayer_perceptron.C   = to_tune(),
  multilayer_perceptron.R   = to_tune(),
  multilayer_perceptron.D   = to_tune(),
  multilayer_perceptron.S   = to_tune(1, 1),

  # REPTree
  reptree.M   = to_tune(1, 64, logscale = TRUE),
  reptree.V   = to_tune(1e-5, 1e-1, logscale = TRUE),
  #FIXME: how to add both?
  #L   = to_tune(1, 1)
  reptree.L   = to_tune(2, 20),
  reptree.P   = to_tune(),

  # IBk
  #IBk.E   = to_tune(),
  #IBk.K   = to_tune(1, 64),
  #IBk.X   = to_tune(),
  #IBk.F   = to_tune(),
  #IBk.I   = to_tune(),

  # RandomForestWeka
  random_forest_weka.I       = to_tune(2, 256, logscale = TRUE),
  random_forest_weka.K       = to_tune(0, 32),
  random_forest_weka.depth   = to_tune(0, 20),

  # RandomTree
  random_tree.M       = to_tune(1, 64, logscale = TRUE),
  #FIXME: K is 0 and 2-32
  random_tree.K       = to_tune(0, 32),
  #FIXME: depth is 0 and 2-20
  random_tree.depth   = to_tune(0, 20),
  #FIXME: N is 0 and 2-5
  random_tree.N       = to_tune(2, 5),
  random_tree.U       = to_tune()
)

tuning_space_classif_autoweka = append(tuning_space_common_autoweka, list(
  # J48
  J48.O     = to_tune(),
  J48.U     = to_tune(),
  J48.B     = to_tune(),
  J48.J     = to_tune(),
  J48.A     = to_tune(),
  J48.S     = to_tune(),
  J48.M     = to_tune(1, 64, logscale = TRUE),
  J48.C     = to_tune(0, 1),
  J48.R     = to_tune(), # Otherwise tuning C, U and N is not possible

  # LMT
  LMT.B     = to_tune(),
  LMT.R     = to_tune(),
  LMT.C     = to_tune(),
  LMT.P     = to_tune(),
  LMT.M     = to_tune(1, 64, logscale = TRUE),
  LMT.W     = to_tune(0, 1),
  LMT.A     = to_tune(),

  # PART
  PART.N     = to_tune(2, 5),
  PART.M     = to_tune(1, 64, logscale = TRUE),
  PART.R     = to_tune(),
  PART.B     = to_tune(),

  # SMO
  smo.C       = to_tune(0.5, 1.5),
  smo.N       = to_tune(levels = c("0", "1", "2")),
  smo.M       = to_tune(),
  smo.K       = to_tune(levels = c("NormalizedPolyKernel", "PolyKernel", "Puk", "RBFKernel")),
  smo.E_poly  = to_tune(0.2, 5),
  smo.L_poly  = to_tune(),

  # BayesNet
  bayes_net.D   = to_tune(),
  bayes_net.Q   = to_tune(levels = c("local.K2", "local.HillClimber", "local.LAGDHillClimber",
                                    "local.SimulatedAnnealing", "local.TabuSearch", "local.TAN")),

  # JRip
  JRip.N   = to_tune(1, 5),
  JRip.E   = to_tune(),
  JRip.P   = to_tune(),
  JRip.O   = to_tune(1, 5),

  # SimpleLogistic
  simple_logistic.S   = to_tune(),
  simple_logistic.W   = to_tune(0, 1),
  simple_logistic.A   = to_tune(),

  # VotedPerceptron
  voted_perceptron.I   = to_tune(1, 10),
  voted_perceptron.M   = to_tune(5000, 50000, logscale = TRUE),
  voted_perceptron.E   = to_tune(0.2, 5),

  # Logistic
  logistic.R = to_tune(1e-12, 10, logscale = TRUE),

  # OneR
  OneR.B = to_tune(1, 32, logscale = TRUE)
))

tuning_space_regr_autoweka = append(tuning_space_common_autoweka, list(
  # Gaussian Processes
  gaussian_processes.L       = to_tune(0.0001, 1, logscale = TRUE),
  gaussian_processes.N       = to_tune(levels = c("0", "1", "2")),
  gaussian_processes.K       = to_tune(levels = c("supportVector.NormalizedPolyKernel", "supportVector.PolyKernel",
                                                 "supportVector.Puk", "supportVector.RBFKernel")),
  gaussian_processes.E_poly  = to_tune(0.2, 5),
  gaussian_processes.L_poly  = to_tune(),

  # M5P
  m5p.N       = to_tune(),
  m5p.M       = to_tune(1, 64, logscale = TRUE),
  m5p.U       = to_tune(),
  m5p.R       = to_tune(),

  # LinearRegression
  linear_regression.S   = to_tune(levels = c("0", "1", "2")),
  linear_regression.C   = to_tune(),
  linear_regression.R   = to_tune(1e-7, 10, logscale = TRUE),

  # M5Rules
  M5Rules.N   = to_tune(),
  M5Rules.M   = to_tune(1, 64, logscale = TRUE),
  M5Rules.U   = to_tune(),
  M5Rules.R   = to_tune(),

  # SMOreg
  smo_reg.C              = to_tune(0.5, 1.5),
  smo_reg.N              = to_tune(levels = c("0", "1", "2")),
  smo_reg.I              = to_tune(levels = c("RegSMOImproved")),
  smo_reg.V_improved     = to_tune(),
  smo_reg.K              = to_tune(levels = c("NormalizedPolyKernel", "PolyKernel", "Puk", "RBFKernel")),
  smo_reg.E_poly         = to_tune(0.2, 5),
  smo_reg.L_poly         = to_tune()
))

#' @title Classification Auto-WEKA Learner
#'
#' @description
#' Classification Auto-WEKA learner.
#'
#' @param id (`character(1)`)\cr
#'   Identifier for the new instance.
#' @param resampling ([mlr3::Resampling]).
#' @param measure ([mlr3::Measure]).
#' @param terminator ([bbotk::Terminator]).
#' @param callbacks (list of [mlr3tuning::CallbackTuning]).
#'
#' @export
LearnerClassifAutoWEKA = R6Class("LearnerClassifAutoWEKA",
  inherit = LearnerAutoWEKA,
  public = list(

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(
      id = "classif.autoweka",
      resampling = rsmp("cv", folds = 3),
      measure = msr("classif.ce"),
      terminator = trm("evals", n_evals = 100L),
      callbacks = list()
      ){

      learner_ids = c("J48", "decision_table", "kstar", "LMT", "PART", "bayes_net", "JRip", "simple_logistic",
        "voted_perceptron", "sgd", "logistic", "OneR", "multilayer_perceptron", "reptree", "random_forest_weka",
        "random_tree", "smo") #IBk

      super$initialize(
        id = id,
        task_type = "classif",
        learner_ids = learner_ids,
        tuning_space = tuning_space_classif_autoweka,
        resampling = resampling,
        measure = measure,
        terminator = terminator,
        callbacks = callbacks)
    }
  )
)

#' @title Regression Auto-WEKA Learner
#'
#' @description
#' Regression Auto-WEKA learner.
#'
#' @param id (`character(1)`)\cr
#'   Identifier for the new instance.
#' @param resampling ([mlr3::Resampling]).
#' @param measure ([mlr3::Measure]).
#' @param terminator ([bbotk::Terminator]).
#' @param callbacks (list of [mlr3tuning::CallbackTuning]).
#'
#' @export
LearnerRegrAutoWEKA = R6Class("LearnerRegrAutoWEKA",
  inherit = LearnerAutoWEKA,
  public = list(

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(
      id = "regr.autoweka",
      resampling = rsmp("cv", folds = 3),
      measure = msr("regr.rmse"),
      terminator = trm("evals", n_evals = 100L),
      callbacks = list()
      ){

      learner_ids = c("decision_table", "m5p", "kstar", "linear_regression", "sgd",
        "multilayer_perceptron", "reptree", "M5Rules", "random_forest_weka", "random_tree",
        "gaussian_processes","smo_reg") #"IBk"

      super$initialize(
        id = id,
        task_type = "regr",
        learner_ids = learner_ids,
        tuning_space = tuning_space_regr_autoweka,
        resampling = resampling,
        measure = measure,
        terminator = terminator,
        callbacks = callbacks)
    }
  )
)
