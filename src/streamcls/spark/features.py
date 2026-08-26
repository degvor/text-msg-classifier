"""The feature pipeline.

Deliberately built from **native Spark ML stages** rather than Spark NLP.

The previous pipeline used Spark NLP annotators, and that was the single worst
dependency problem in the project: the runtime images baked in a hand-curl'ed
``spark-nlp 4.4.0`` jar, while the trainer additionally asked Ivy for ``5.1.3``
via ``spark.jars.packages`` -- so the training container had *both* on the
classpath at once and which one won was classloader-order dependent, while the
serving container declared no Spark NLP at all and deserialised 5.1.3-written
annotators with 4.4.0. For a bag-of-words classifier, ``RegexTokenizer`` and
``StopWordsRemover`` do the same job with none of that.

Three preprocessing defects are fixed here as well; each was verified in the
saved model's own metadata.

* ``Normalizer`` had ``lowercase: false``, so ``Tech`` and ``tech`` were
  different features. Combined with the rest, this is why the headline
  ``Tech stocks lead market rally`` produced an all-zero feature vector.
* ``cleanupPatterns: ["[^\\pL+]"]`` deleted every digit, so ``2024`` and ``$2``
  vanished. (The ``+`` inside that negated class is a literal plus, almost
  certainly not what was intended.)
* The English stopword list was applied to a bilingual corpus, leaving Ukrainian
  function words present in all 50 samples of a class -- high-document-frequency
  features that encoded template structure rather than topic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.ml import Pipeline

__all__ = ["ALGORITHMS", "FEATURE_COLUMN", "LABEL_COLUMN", "TEXT_COLUMN", "build_pipeline"]

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
FEATURE_COLUMN = "features"
#: Kept on the output so drift detection can sum it per batch. It is the exact
#: feature space the classifier consumes, which is what makes PSI over it mean
#: "the input the model sees has shifted".
RAW_FEATURE_COLUMN = "raw_features"

#: Algorithms available to the comparison sweep. LogReg is the default because
#: it is the standard AG News baseline the thesis compares against.
ALGORITHMS = ("logreg", "naive_bayes", "random_forest", "linear_svc")


def _classifier(algorithm: str, params: dict[str, Any]):
    from pyspark.ml.classification import (
        LinearSVC,
        LogisticRegression,
        NaiveBayes,
        OneVsRest,
        RandomForestClassifier,
    )

    if algorithm == "logreg":
        return LogisticRegression(
            featuresCol=FEATURE_COLUMN,
            labelCol=LABEL_COLUMN,
            # The old model was saved with regParam=0.0 -- no regularisation at
            # all -- on a linearly separable problem, and maxIter=20, which is
            # a truncation rather than a convergence criterion.
            regParam=params.get("reg_param", 0.05),
            elasticNetParam=params.get("elastic_net_param", 0.0),
            maxIter=params.get("max_iter", 100),
            tol=params.get("tol", 1e-6),
        )
    if algorithm == "naive_bayes":
        return NaiveBayes(
            featuresCol=FEATURE_COLUMN,
            labelCol=LABEL_COLUMN,
            smoothing=params.get("smoothing", 1.0),
        )
    if algorithm == "random_forest":
        return RandomForestClassifier(
            featuresCol=FEATURE_COLUMN,
            labelCol=LABEL_COLUMN,
            numTrees=params.get("num_trees", 100),
            maxDepth=params.get("max_depth", 12),
            seed=params.get("seed", 42),
        )
    if algorithm == "linear_svc":
        # Binary-only, so wrap it. No probability column, which is why it is not
        # the default: the drift detectors use prediction confidence.
        return OneVsRest(
            classifier=LinearSVC(
                featuresCol=FEATURE_COLUMN,
                labelCol=LABEL_COLUMN,
                regParam=params.get("reg_param", 0.05),
                maxIter=params.get("max_iter", 50),
            ),
            labelCol=LABEL_COLUMN,
            featuresCol=FEATURE_COLUMN,
        )
    raise ValueError(f"unknown algorithm {algorithm!r}; expected one of {ALGORITHMS}")


def build_pipeline(algorithm: str = "logreg", **params: Any) -> Pipeline:
    """Text in, prediction out. No label indexer: label ids are pinned.

    Dropping ``StringIndexer`` removes the failure where the integer for each
    class depended on the training class balance, so a retrain could silently
    renumber every class and retroactively change the meaning of every stored
    prediction.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import IDF, CountVectorizer, RegexTokenizer, StopWordsRemover

    tokenizer = RegexTokenizer(
        inputCol=TEXT_COLUMN,
        outputCol="tokens",
        # Split on non-word runs. Keeps digits (2024, $2 -> "2") and drops the
        # punctuation the perturbation engine injects.
        pattern=r"\W+",
        toLowercase=True,
        minTokenLength=2,
    )
    stopwords = StopWordsRemover(inputCol="tokens", outputCol="terms")
    counts = CountVectorizer(
        inputCol="terms",
        outputCol=RAW_FEATURE_COLUMN,
        vocabSize=params.get("vocab_size", 20_000),
        minDF=params.get("min_df", 3.0),
    )
    idf = IDF(inputCol=RAW_FEATURE_COLUMN, outputCol=FEATURE_COLUMN, minDocFreq=0)

    return Pipeline(stages=[tokenizer, stopwords, counts, idf, _classifier(algorithm, params)])
