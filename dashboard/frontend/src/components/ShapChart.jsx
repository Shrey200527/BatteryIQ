function ShapChart({
  featureContributions = [],
  loading = false,
  error = "",
}) {
      if (loading) {
    return (
      <div className="shap-card">
        <div className="shap-header">
          <div>
            <span className="shap-label">MODEL FEATURE CONTRIBUTION</span>
            <h3>SHAP Feature Contribution</h3>
          </div>

          <span className="shap-status pending">PROCESSING</span>
        </div>

        <div className="shap-empty">
          <p>Generating model feature contribution data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="shap-card">
        <div className="shap-header">
          <div>
            <span className="shap-label">MODEL FEATURE CONTRIBUTION</span>
            <h3>SHAP Feature Contribution</h3>
          </div>

          <span className="shap-status unavailable">UNAVAILABLE</span>
        </div>

        <div className="shap-empty">
          <p>{error}</p>
        </div>
      </div>
    );
  }
  // --------------------------------
  // No SHAP data available
  // --------------------------------

  if (!featureContributions || featureContributions.length === 0) {
    return (
      <div className="shap-card">

        <div className="shap-header">
          <div>
            <span className="shap-label">
              MODEL FEATURE CONTRIBUTION
            </span>

            <h3>SHAP Feature Contribution</h3>
          </div>

          <span className="shap-status pending">
            PENDING
          </span>
        </div>

        <div className="shap-empty">
          <div className="placeholder-icon">
            📊
          </div>

          <p>
            Feature contribution data will be displayed
            after the approved SHAP endpoint provides
            a prediction explanation.
          </p>
        </div>

      </div>
    );
  }

  // --------------------------------
  // SHAP data available
  // --------------------------------

  return (
    <div className="shap-card">

      <div className="shap-header">
        <div>
          <span className="shap-label">
            MODEL FEATURE CONTRIBUTION
          </span>

          <h3>SHAP Feature Contribution</h3>
        </div>

        <span className="shap-status available">
          AVAILABLE
        </span>
      </div>

      <div className="shap-list">

        {featureContributions.map((feature, index) => {

          const featureName =
            feature.feature ||
            feature.name ||
            `Feature ${index + 1}`;

          const contribution =
            Number(feature.contribution ?? feature.shap_value ?? 0);

          return (
            <div
              className="shap-row"
              key={`${featureName}-${index}`}
            >

              <div className="shap-feature-name">
                {featureName}
              </div>

              <div className="shap-bar-container">

                <div
                  className="shap-bar"
                  style={{
                    width: `${Math.min(
                      Math.abs(contribution) * 100,
                      100
                    )}%`,
                  }}
                />

              </div>

              <div className="shap-value">
                {contribution > 0 ? "+" : ""}
                {contribution.toFixed(4)}
              </div>

            </div>
          );
        })}

      </div>

      <p className="shap-note">
        SHAP values represent model feature contributions
        for the prediction. They should not be interpreted
        as causal effects.
      </p>

    </div>
  );
}

export default ShapChart;