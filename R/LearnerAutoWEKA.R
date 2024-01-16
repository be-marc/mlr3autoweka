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
#'   List of learner ids.
#' @param tuning_space ([paradox::ParamSet])\cr
#'   Tuning space of the graph learner.
#' @param fallback_learner ([Learner])\cr
#'  Fallback learner.
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

    #' @field fallback_learner ([Learner]).
    fallback_learner = NULL,

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(id, task_type, learner_ids, tuning_space, fallback_learner, resampling, measure, terminator, callbacks = list()) {
      assert_choice(task_type, mlr_reflections$task_types$type)
      self$learner_ids = assert_character(learner_ids)
      self$tuning_space = assert_param_set(tuning_space)
      self$fallback_learner = assert_learner(fallback_learner)


      self$resampling = assert_resampling(resampling)
      self$measure = assert_measure(measure)
      self$terminator = assert_terminator(terminator)
      self$callbacks = assert_list(as_callbacks(callbacks), types = "CallbackTuning")


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

      # initialize mbo tuner
      surrogate = default_surrogate(n_learner = 1, search_space = self$tuning_space, noisy = TRUE)
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
        search_space = self$tuning_space
      )

      auto_tuner$train(task)
      auto_tuner
    },

    .predict = function(task) {
      self$model$predict(task)
    }
  )
)

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
      assert_measure(measure)
      learner_ids = c("J48", "decision_table", "kstar", "LMT", "PART", "bayes_net", "JRip", "simple_logistic",
        "voted_perceptron", "sgd", "logistic", "OneR", "multilayer_perceptron", "reptree", "random_forest_weka",
        "random_tree", "smo", "IBk")

      super$initialize(
        id = id,
        task_type = "classif",
        learner_ids = learner_ids,
        tuning_space = tuning_space_classif_autoweka,
        fallback_learner = lrn("classif.featureless", predict_type = measure$predict_type),
        resampling = resampling,
        measure = measure,
        terminator = terminator,
        callbacks = callbacks)
    }
  )
)

tuning_space_classif_autoweka = ps(
  # Decision Table
  decision_table.S = p_fct(c("BestFirst", "GreedyStepwise"), depends = (branch.selection == "decision_table")),
  decision_table.X = p_int(1, 4, depends = (branch.selection == "decision_table")),
  decision_table.E = p_fct(c("acc", "auc"), depends = (branch.selection == "decision_table")),
  decision_table.I = p_lgl(depends = (branch.selection == "decision_table")),

  # KStar
  kstar.B = p_int(1, 100, depends = (branch.selection == "kstar")),
  kstar.E = p_lgl(depends = (branch.selection == "kstar")),
  kstar.M = p_fct(c("a", "d", "m", "n"), depends = (branch.selection == "kstar")),

  # SGD
  sgd.F = p_fct(c("0", "1"), depends = (branch.selection == "sgd")),
  sgd.L = p_dbl(0.00001, 0.1, logscale = TRUE, depends = (branch.selection == "sgd")),
  sgd.R = p_dbl(1e-12, 10, logscale = TRUE, depends = (branch.selection == "sgd")),
  sgd.N = p_lgl(depends = (branch.selection == "sgd")),
  sgd.M = p_lgl(depends = (branch.selection == "sgd")),

  # MultilayerPerceptron
  multilayer_perceptron.L = p_dbl(0.1, 1, depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.M = p_dbl(0.1, 1, depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.B = p_lgl(depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.H = p_fct(c("a", "i", "o", "t"), depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.C = p_lgl(depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.R = p_lgl(depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.D = p_lgl(depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.S = p_int(1, 1, depends = (branch.selection == "multilayer_perceptron")),

  # REPTree
  reptree.M = p_int(1, 64, logscale = TRUE, depends = (branch.selection == "reptree")),
  reptree.V = p_dbl(1e-5, 1e-1, logscale = TRUE, depends = (branch.selection == "reptree")),
  reptree.L = p_int(2, 20, depends = (reptree.depth_HIDDEN == TRUE && branch.selection == "reptree")),
  reptree.P = p_lgl(depends = (branch.selection == "reptree")),
  reptree.depth_HIDDEN = p_lgl(depends = (branch.selection == "reptree")),

  # IBk
  IBk.weight = p_fct(c("I", "F"), depends = (branch.selection == "IBk")),
  IBk.K = p_int(1, 64, logscale = TRUE, depends = (branch.selection == "IBk")),
  IBk.E = p_lgl(depends = (branch.selection == "IBk")),
  IBk.X = p_lgl(depends = (branch.selection == "IBk")),

  # RandomForestWeka
  random_forest_weka.I = p_int(2, 256, logscale = TRUE, depends = (branch.selection == "random_forest_weka")),
  random_forest_weka.K = p_int(0, 32, depends = (branch.selection == "random_forest_weka")),
  random_forest_weka.depth = p_int(0, 20, depends = (branch.selection == "random_forest_weka")),

  # RandomTree
  random_tree.K = p_int(0, 32, depends = (branch.selection == "random_tree")),
  random_tree.M = p_int(1, 64, logscale = TRUE, depends = (branch.selection == "random_tree")),
  random_tree.depth = p_int(0, 20, depends = (branch.selection == "random_tree")),
  random_tree.N = p_int(2, 5, depends = (branch.selection == "random_tree")),
  random_tree.U = p_lgl(depends = (branch.selection == "random_tree")),

  # J48
  J48.U = p_lgl(depends = (branch.selection == "J48")),
  J48.O = p_lgl(depends = (branch.selection == "J48")),
  J48.C = p_dbl(.Machine$double.eps, 1 - .Machine$double.eps, depends = (branch.selection == "J48" && J48.U == FALSE && J48.R == FALSE)),
  J48.M = p_int(1, 64, logscale = TRUE, depends = (branch.selection == "J48")),
  J48.R = p_lgl(depends = (branch.selection == "J48" && J48.U == FALSE)),
  J48.B = p_lgl(depends = (branch.selection == "J48")),
  J48.S = p_lgl(depends = (branch.selection == "J48" && J48.U == FALSE)),
  J48.L = p_lgl(depends = (branch.selection == "J48")),
  J48.A = p_lgl(depends = (branch.selection == "J48")),
  J48.J = p_lgl(depends = (branch.selection == "J48")),

  # LMT
  LMT.B = p_lgl(depends = (branch.selection == "LMT")),
  LMT.R = p_lgl(depends = (branch.selection == "LMT")),
  LMT.C = p_lgl(depends = (branch.selection == "LMT")),
  LMT.P = p_lgl(depends = (branch.selection == "LMT")),
  LMT.M = p_int(1, 64, logscale = TRUE, depends = (branch.selection == "LMT")),
  LMT.W = p_dbl(0, 1, depends = (branch.selection == "LMT")),
  LMT.A = p_lgl(depends = (branch.selection == "LMT")),

  # PART
  PART.M = p_int(1, 64, logscale = TRUE, depends = (branch.selection == "PART")),
  PART.R = p_lgl(depends = (branch.selection == "PART")),
  PART.N = p_int(2, 5, depends = (branch.selection == "PART" && PART.R == TRUE)),
  PART.B = p_lgl(depends = (branch.selection == "PART")),

  # SMO
  smo.C = p_dbl(0.5, 1.5, depends = (branch.selection == "smo")),
  smo.N = p_fct(c("0", "1", "2"), depends = (branch.selection == "smo")),
  smo.M = p_lgl(depends = (branch.selection == "smo")),
  smo.K = p_fct(c("NormalizedPolyKernel", "PolyKernel", "Puk", "RBFKernel"), depends = (branch.selection == "smo")),
  smo.E_poly = p_dbl(0.2, 5, depends = (smo.K == "PolyKernel" && branch.selection == "smo")),
  smo.L_poly = p_lgl(depends = (smo.K == "PolyKernel" && branch.selection == "smo")),

  # BayesNet
  bayes_net.D = p_lgl(depends = (branch.selection == "bayes_net")),
  bayes_net.Q = p_fct(c("local.K2", "local.HillClimber", "local.LAGDHillClimber", "local.SimulatedAnnealing", "local.TabuSearch", "local.TAN"), depends = (branch.selection == "bayes_net")),

  # JRip
  JRip.N = p_dbl(1, 5, depends = (branch.selection == "JRip")),
  JRip.O = p_int(1, 5, depends = (branch.selection == "JRip")),
  JRip.E = p_lgl(depends = (branch.selection == "JRip")),
  JRip.P = p_lgl(depends = (branch.selection == "JRip")),

  # SimpleLogistic
  simple_logistic.S = p_lgl(depends = (branch.selection == "simple_logistic")),
  simple_logistic.W = p_dbl(0, 1, depends = (branch.selection == "simple_logistic")),
  simple_logistic.A = p_lgl(depends = (branch.selection == "simple_logistic")),

  # VotedPerceptron
  voted_perceptron.I = p_int(1, 10, depends = (branch.selection == "voted_perceptron")),
  voted_perceptron.E = p_dbl(0.2, 5, depends = (branch.selection == "voted_perceptron")),
  voted_perceptron.M = p_int(5000, 50000, logscale = TRUE, depends = (branch.selection == "voted_perceptron")),

  # Logistic
  logistic.R = p_dbl(1e-12, 10, logscale = TRUE, depends = (branch.selection == "logistic")),

  # OneR
  OneR.B = p_int(1, 32, logscale = TRUE, depends = (branch.selection == "OneR")),

  # Branch
  branch.selection = p_fct(c("J48", "decision_table", "kstar", "LMT", "PART", "bayes_net", "JRip", "simple_logistic",
                             "voted_perceptron", "sgd", "logistic", "OneR", "multilayer_perceptron", "reptree",
                             "random_forest_weka", "random_tree", "smo", "IBk")),

  .extra_trafo = function(x, param_set)  {
    if (isFALSE(x$reptree.depth_HIDDEN)) {
      x$reptree.L = -1
    }
    x$reptree.depth_HIDDEN = NULL
    x
  }
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

      learner_ids = c("decision_table", "m5p", "kstar", "linear_regression", "sgd", "multilayer_perceptron", "reptree",
        "M5Rules", "random_forest_weka", "random_tree", "gaussian_processes","smo_reg", "IBk")

      super$initialize(
        id = id,
        task_type = "regr",
        learner_ids = learner_ids,
        tuning_space = tuning_space_regr_autoweka,
        fallback_learner = lrn("regr.featureless"),
        resampling = resampling,
        measure = measure,
        terminator = terminator,
        callbacks = callbacks)
    }
  )
)

tuning_space_regr_autoweka = ps(

  # Decision Table
  decision_table.S = p_fct(c("BestFirst", "GreedyStepwise"), depends = (branch.selection == "decision_table")),
  decision_table.X = p_int(1, 4, depends = (branch.selection == "decision_table")),
  decision_table.E = p_fct(c("rmse","mae"), depends = (branch.selection == "decision_table")),
  decision_table.I = p_lgl(depends = (branch.selection == "decision_table")),

  # KStar
  kstar.B = p_int(1, 100, depends = (branch.selection == "kstar")),
  kstar.E = p_lgl(depends = (branch.selection == "kstar")),
  kstar.M = p_fct(c("a", "d", "m", "n"), depends = (branch.selection == "kstar")),

  # SGD
  sgd.F = p_fct(c("2", "3", "4"), depends = (branch.selection == "sgd")),
  sgd.L = p_dbl(0.00001, 0.1, logscale = TRUE, depends = (branch.selection == "sgd")),
  sgd.R = p_dbl(1e-12, 10, logscale = TRUE, depends = (branch.selection == "sgd")),
  sgd.N = p_lgl(depends = (branch.selection == "sgd")),
  sgd.M = p_lgl(depends = (branch.selection == "sgd")),

  # MultilayerPerceptron
  multilayer_perceptron.L = p_dbl(0.1, 1, depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.M = p_dbl(0.1, 1, depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.B = p_lgl(depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.H = p_fct(c("a", "i", "o", "t"), depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.C = p_lgl(depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.R = p_lgl(depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.D = p_lgl(depends = (branch.selection == "multilayer_perceptron")),
  multilayer_perceptron.S = p_int(1, 1, depends = (branch.selection == "multilayer_perceptron")),

  # REPTree
  reptree.M = p_int(1, 64, logscale = TRUE, depends = (branch.selection == "reptree")),
  reptree.V = p_dbl(1e-5, 1e-1, logscale = TRUE, depends = (branch.selection == "reptree")),
  reptree.L = p_int(2, 20, depends = (reptree.depth_HIDDEN == TRUE && branch.selection == "reptree")),
  reptree.P = p_lgl(depends = (branch.selection == "reptree")),
  reptree.depth_HIDDEN = p_lgl(depends = (branch.selection == "reptree")),

  # IBk
  IBk.weight = p_fct(c("I", "F"), depends = (branch.selection == "IBk")),
  IBk.K = p_int(1, 64, logscale = TRUE, depends = (branch.selection == "IBk")),
  IBk.E = p_lgl(depends = (branch.selection == "IBk")),
  IBk.X = p_lgl(depends = (branch.selection == "IBk")),

  # RandomForestWeka
  random_forest_weka.I = p_int(2, 256, logscale = TRUE, depends = (branch.selection == "random_forest_weka")),
  random_forest_weka.K = p_int(0, 32, depends = (branch.selection == "random_forest_weka")),
  random_forest_weka.depth = p_int(0, 20, depends = (branch.selection == "random_forest_weka")),

  # RandomTree
  random_tree.K = p_int(0, 32, depends = (branch.selection == "random_tree")),
  random_tree.M = p_int(1, 64, logscale = TRUE, depends = (branch.selection == "random_tree")),
  random_tree.depth = p_int(0, 20, depends = (branch.selection == "random_tree")),
  random_tree.N = p_int(2, 5, depends = (branch.selection == "random_tree")),
  random_tree.U = p_lgl(depends = (branch.selection == "random_tree")),

  # Gaussian Processes
  gaussian_processes.L = p_dbl(0.0001, 1, logscale = TRUE, depends = (branch.selection == "gaussian_processes")),
  gaussian_processes.N = p_fct(c("0", "1", "2")),
  gaussian_processes.K = p_fct(c("supportVector.NormalizedPolyKernel", "supportVector.PolyKernel", "supportVector.Puk", "supportVector.RBFKernel")),
  gaussian_processes.E_poly = p_dbl(0.2, 5, depends = (gaussian_processes.K == "supportVector.PolyKernel")),
  gaussian_processes.L_poly = p_lgl(depends = (gaussian_processes.K == "supportVector.PolyKernel")),

  # M5P
  m5p.N = p_lgl(depends = (branch.selection == "m5p")),
  m5p.U = p_lgl(depends = (branch.selection == "m5p")),
  m5p.R = p_lgl(depends = (branch.selection == "m5p")),
  m5p.M = p_int(1, 64, logscale = TRUE, depends = (branch.selection == "m5p")),

  # LinearRegression
  linear_regression.S = p_fct(c("0", "1", "2"), depends = (branch.selection == "linear_regression")),
  linear_regression.C = p_lgl(depends = (branch.selection == "linear_regression")),
  linear_regression.R = p_dbl(1e-7, 10, logscale = TRUE, depends = (branch.selection == "linear_regression")),

  # M5Rules
  M5Rules.N = p_lgl(depends = (branch.selection == "M5Rules")),
  M5Rules.U = p_lgl(depends = (branch.selection == "M5Rules")),
  M5Rules.R = p_lgl(depends = (branch.selection == "M5Rules")),
  M5Rules.M = p_int(1, 64, logscale = TRUE, depends = (branch.selection == "M5Rules")),

  # SMOreg
  smo_reg.C = p_dbl(0.5, 1.5, depends = (branch.selection == "smo_reg")),
  smo_reg.N = p_fct(c("0", "1", "2"), depends = (branch.selection == "smo_reg")),
  smo_reg.I = p_fct(c("RegSMOImproved"), depends = (branch.selection == "smo_reg")),
  smo_reg.K = p_fct(c("NormalizedPolyKernel", "PolyKernel", "Puk", "RBFKernel"), depends = (branch.selection == "smo_reg")),
  smo_reg.V_improved = p_lgl(depends = (branch.selection == "smo_reg")),
  smo_reg.E_poly = p_dbl(0.2, 5, depends = (smo_reg.K == "PolyKernel" && branch.selection == "smo_reg")),
  smo_reg.L_poly = p_lgl(depends = (smo_reg.K == "PolyKernel" && branch.selection == "smo_reg")),

  # Branch
  branch.selection = p_fct(c("decision_table", "m5p", "kstar", "linear_regression", "sgd", "multilayer_perceptron",
    "reptree", "M5Rules", "random_forest_weka", "random_tree", "gaussian_processes","smo_reg", "IBk")),

  .extra_trafo = function(x, param_set)  {
    if (isFALSE(x$reptree.depth_HIDDEN)) {
      x$reptree.L = -1
    }
    x$reptree.depth_HIDDEN = NULL
    x
  }
)
