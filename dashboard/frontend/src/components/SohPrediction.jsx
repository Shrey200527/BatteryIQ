function SohPrediction({
  prediction = null,
  loading = false,
  error = "",
}) {
  if (loading) {
    return (
      <div className="prediction-card">
        <div className="prediction-header">
          <div>
            <span className="prediction-label">SOH PREDICTION</span>
            <h3>State of Health</h3>
          </div>

          <span className="prediction-status loading">
            PROCESSING
          </span>
        </div>

        <p>Generating SOH prediction...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="prediction-card">
        <div className="prediction-header">
          <div>
            <span className="prediction-label">SOH PREDICTION</span>
            <h3>State of Health</h3>
          </div>

          <span className="prediction-status error">
            ERROR
          </span>
        </div>

        <p>{error}</p>
      </div>
    );
  }

  if (prediction === null || prediction === undefined) {
    return (
      <div className="prediction-card">
        <div className="prediction-header">
          <div>
            <span className="prediction-label">SOH PREDICTION</span>
            <h3>State of Health</h3>
          </div>

          <span className="prediction-status pending">
            PENDING
          </span>
        </div>

        <p>
          SOH prediction will be displayed when the required
          cycle-level features are available.
        </p>
      </div>
    );
  }

  return (
    <div className="prediction-card">
      <div className="prediction-header">
        <div>
          <span className="prediction-label">SOH PREDICTION</span>
          <h3>State of Health</h3>
        </div>

        <span className="prediction-status available">
          AVAILABLE
        </span>
      </div>

      <div className="rul-value">
        {Number(prediction).toFixed(2)}%
      </div>

      <p>
        SOH prediction returned by the approved BatteryIQ model.
      </p>
    </div>
  );
}

export default SohPrediction;