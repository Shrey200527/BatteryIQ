function RulPrediction({
  batteryId,
  rulValid,
  prediction = null,
  loading = false,
  error = "",
}) {
  // -----------------------------
  // RUL unavailable
  // -----------------------------

  if (!rulValid) {
    return (
      <div className="prediction-card">

        <div className="prediction-header">
          <div>
            <span className="prediction-label">
              REMAINING USEFUL LIFE
            </span>

            <h3>Unavailable</h3>
          </div>

          <span className="prediction-status unavailable">
            UNAVAILABLE
          </span>
        </div>

        <p>
          RUL prediction is not available for {batteryId}.
          This battery is right-censored according to the
          BatteryIQ ML contract.
        </p>

      </div>
    );
  }

  // -----------------------------
  // Loading
  // -----------------------------

  if (loading) {
    return (
      <div className="prediction-card">

        <div className="prediction-header">
          <div>
            <span className="prediction-label">
              REMAINING USEFUL LIFE
            </span>

            <h3>Calculating...</h3>
          </div>

          <span className="prediction-status loading">
            PROCESSING
          </span>
        </div>

        <p>
          BatteryIQ is requesting an RUL prediction from
          the approved prediction model.
        </p>

      </div>
    );
  }

  // -----------------------------
  // Error
  // -----------------------------

  if (error) {
    return (
      <div className="prediction-card">

        <div className="prediction-header">
          <div>
            <span className="prediction-label">
              REMAINING USEFUL LIFE
            </span>

            <h3>Prediction unavailable</h3>
          </div>

          <span className="prediction-status error">
            ERROR
          </span>
        </div>

        <p>
          {error}
        </p>

      </div>
    );
  }

  // -----------------------------
  // Prediction available
  // -----------------------------

  if (prediction !== null) {
    return (
      <div className="prediction-card">

        <div className="prediction-header">
          <div>
            <span className="prediction-label">
              REMAINING USEFUL LIFE
            </span>

            <h3>RUL Prediction</h3>
          </div>

          <span className="prediction-status available">
            AVAILABLE
          </span>
        </div>

        <div className="rul-value">
          {prediction}
        </div>

        <p>
          Estimated remaining useful life in cycles.
        </p>

      </div>
    );
  }

  // -----------------------------
  // Default / waiting state
  // -----------------------------

  return (
    <div className="prediction-card">

      <div className="prediction-header">
        <div>
          <span className="prediction-label">
            REMAINING USEFUL LIFE
          </span>

          <h3>Prediction Ready</h3>
        </div>

        <span className="prediction-status pending">
          READY
        </span>
      </div>

      <p>
        An approved RUL prediction can be requested when
        the required cycle features are available.
      </p>

    </div>
  );
}

export default RulPrediction;