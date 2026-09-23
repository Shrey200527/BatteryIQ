function ConditionStatus({ condition = null, status = "thresholds_pending" }) {
  // --------------------------------
  // Approved condition available
  // --------------------------------

  if (status === "available" && condition) {
    return (
      <div className="condition-card">
        <div className="condition-header">
          <div>
            <span className="condition-label">
              BATTERY CONDITION
            </span>

            <h3>{condition}</h3>
          </div>

          <span className="condition-status available">
            AVAILABLE
          </span>
        </div>

        <p>
          Battery condition returned by the approved
          BatteryIQ classification system.
        </p>
      </div>
    );
  }

  // --------------------------------
  // Thresholds are not approved yet
  // --------------------------------

  if (status === "thresholds_pending") {
    return (
      <div className="condition-card">
        <div className="condition-header">
          <div>
            <span className="condition-label">
              BATTERY CONDITION
            </span>

            <h3>Thresholds Pending</h3>
          </div>

          <span className="condition-status pending">
            PENDING
          </span>
        </div>

        <p>
          Classification thresholds have not yet been
          approved by the team.
        </p>
      </div>
    );
  }

  // --------------------------------
  // Error / unavailable state
  // --------------------------------

  return (
    <div className="condition-card">
      <div className="condition-header">
        <div>
          <span className="condition-label">
            BATTERY CONDITION
          </span>

          <h3>Unavailable</h3>
        </div>

        <span className="condition-status unavailable">
          UNAVAILABLE
        </span>
      </div>

      <p>
        Battery condition is currently unavailable.
      </p>
    </div>
  );
}

export default ConditionStatus;