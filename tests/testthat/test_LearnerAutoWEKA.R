test_that("classification graph is constructed", {
  skip_on_ci()

  learner_ids = c("J48", "decision_table", "kstar", "LMT", "PART", "bayes_net", "JRip", "simple_logistic",
                  "voted_perceptron", "sgd", "logistic", "OneR", "multilayer_perceptron", "reptree",
                  "random_forest_weka", "random_tree", "smo", "IBk")

  graph = get_branch_pipeline("classif", learner_ids)
  expect_class(graph, "Graph")
})

test_that("classification search space can be set", {
  skip_on_ci()

  learner_ids = c("J48", "decision_table", "kstar", "LMT", "PART", "bayes_net", "JRip", "simple_logistic",
                  "voted_perceptron", "sgd", "logistic", "OneR", "multilayer_perceptron", "reptree",
                  "random_forest_weka", "random_tree", "smo", "IBk")

  graph = get_branch_pipeline("classif", learner_ids)
  learner = as_learner(graph)

  design = generate_design_random(tuning_space_classif_autoweka, 10000)$data
  xss = transform_xdt_to_xss(design, tuning_space_classif_autoweka)

  walk(xss, function(xs) {
    learner$param_set$values = list()
    expect_class(learner$param_set$set_values(.values = xs), "ParamSet")
  })
})

test_that("regression search space can be set", {
  skip_on_ci()

  learner_ids = c("decision_table", "m5p", "kstar", "linear_regression", "sgd",
                  "multilayer_perceptron", "reptree", "M5Rules", "random_forest_weka",
                  "random_tree", "gaussian_processes","smo_reg", "IBk")

  graph = get_branch_pipeline("regr", learner_ids)
  learner = as_learner(graph)

  design = generate_design_random(tuning_space_regr_autoweka,  10000)$data
  xss = transform_xdt_to_xss(design, tuning_space_regr_autoweka)

  walk(xss, function(xs) {
    learner$param_set$values = list()
    expect_class(learner$param_set$set_values(.values = xs), "ParamSet")
  })
})

test_that("LearnerClassifAutoWEKA train works", {
  skip_on_ci()

  task = tsk("sonar")
  resampling = rsmp("holdout")
  measure = msr("classif.ce")
  terminator = trm("run_time", secs = 10)
  learner = LearnerClassifAutoWEKA$new(
    resampling = resampling,
    measure = measure,
    terminator = terminator)

  expect_names(learner$packages, must.include = "RWeka")

  expect_class(learner$train(task), "LearnerClassifAutoWEKA")
  expect_class(learner$model, "AutoTuner")

  expect_prediction(learner$predict(task))
})

test_that("LearnerRegrAutoWEKA train works", {
  skip_on_ci()

  task = tsk("mtcars")
  resampling = rsmp("holdout")
  measure = msr("regr.mse")
  terminator = trm("run_time", secs = 30)
  learner = LearnerRegrAutoWEKA$new(
    resampling = resampling,
    measure = measure,
    terminator = terminator)

  expect_names(learner$packages, must.include = "RWeka")

  expect_class(learner$train(task), "LearnerRegrAutoWEKA")
  expect_class(learner$model, "AutoTuner")

  expect_prediction(learner$predict(task))
})

test_that("LearnerClassifAutoWEKA resample works", {
  skip_on_ci()

  task = tsk("sonar")
  resampling = rsmp("holdout")
  measure = msr("classif.ce")
  terminator = trm("run_time", secs = 30)
  learner = LearnerClassifAutoWEKA$new(
    resampling = resampling,
    measure = measure,
    terminator = terminator)

  expect_resample_result(resample(task, learner, rsmp("holdout")))
})

test_that("LearnerClassifAutoWEKA train works with parallelization", {
  skip_on_ci()

  future::plan("multisession", workers = 2)

  task = tsk("sonar")
  resampling = rsmp("holdout")
  measure = msr("classif.ce")
  terminator = trm("run_time", secs = 10)
  learner = LearnerClassifAutoWEKA$new(
    resampling = resampling,
    measure = measure,
    terminator = terminator)

  expect_class(learner$train(task), "LearnerClassifAutoWEKA")
  expect_class(learner$model, "AutoTuner")
})
