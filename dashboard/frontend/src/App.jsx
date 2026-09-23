import { useEffect, useState } from "react";
import "./App.css";
import {
  getBatteries,
  getBatteryInfo,
  getBatteryHistory,
  getBatteryCondition,
  getShapExplanation,
} from "./services/api";
import KpiCard from "./components/KpiCard";
import BatteryChart from "./components/BatteryChart";
import RulPrediction from "./components/RulPrediction";
import ConditionStatus from "./components/ConditionStatus";
import ShapChart from "./components/ShapChart";
import SohPrediction from "./components/SohPrediction";

function App() {
  // -----------------------------
  // State
  // -----------------------------

  const [batteries, setBatteries] = useState([]);
  const [selectedBattery, setSelectedBattery] = useState("");

  const [batteryInfo, setBatteryInfo] = useState(null);

  const [loading, setLoading] = useState(true);
  const [infoLoading, setInfoLoading] = useState(false);

  const [rulPrediction, setRulPrediction] = useState(null);
const [rulLoading, setRulLoading] = useState(false);
const [rulError, setRulError] = useState("");

const [featureContributions, setFeatureContributions] = useState([]);
const [shapLoading, setShapLoading] = useState(false);
const [shapError, setShapError] = useState("");
const [batteryHistory, setBatteryHistory] = useState(null);
const [measurementData, setMeasurementData] = useState([]);

const [condition, setCondition] = useState(null);
const [conditionStatus, setConditionStatus] = useState(
  "thresholds_pending"
);
const [conditionLoading, setConditionLoading] = useState(false);

const [sohPrediction, setSohPrediction] = useState(null);
const [sohLoading, setSohLoading] = useState(false);
const [sohError, setSohError] = useState("");

  const [error, setError] = useState("");

  // -----------------------------
  // Load battery list
  // Runs once when the page loads
  // -----------------------------

  useEffect(() => {
    getBatteries()
      .then((data) => {
        console.log("Battery list received:", data);

        setBatteries(data.batteries || []);

        // Automatically select the first battery
        if (data.batteries && data.batteries.length > 0) {
          setSelectedBattery(data.batteries[0].battery_id);
        }
      })
      .catch((err) => {
        console.error("Battery list error:", err);
        setError("Unable to connect to BatteryIQ backend.");
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);


  // -----------------------------
  // Load information for selected battery
  // Runs whenever selectedBattery changes
  // -----------------------------

  useEffect(() => {
  if (!selectedBattery) {
    return;
  }

  setInfoLoading(true);
  setError("");

  // Clear predictions from the previously selected battery
  setSohPrediction(null);
  setSohLoading(false);
  setSohError("");

  setRulPrediction(null);
  setRulLoading(false);
  setRulError("");

  setFeatureContributions([]);
  setShapLoading(false);
  setShapError("");

  getBatteryInfo(selectedBattery)
    .then((data) => {
      console.log("Selected battery information:", data);

      setBatteryInfo(data);
    })
    .catch((err) => {
      console.error("Battery information error:", err);

      setBatteryInfo(null);
      setError("Unable to load battery information.");
    })
    .finally(() => {
      setInfoLoading(false);
    });
}, [selectedBattery]);

  useEffect(() => {
  if (!batteryInfo?.soh_range) {
    return;
  }

  const soh = batteryInfo.soh_range.max;

  setConditionLoading(true);

  getBatteryCondition(soh)
    .then((data) => {
      setCondition(data.condition);
      setConditionStatus(data.status);
    })
    .catch((err) => {
      console.error("Condition error:", err);

      setCondition(null);
      setConditionStatus("unavailable");
    })
    .finally(() => {
      setConditionLoading(false);
    });
}, [batteryInfo]);

useEffect(() => {
  if (!selectedBattery) {
    return;
  }

  getBatteryHistory(selectedBattery)
    .then((data) => {
      console.log("Selected battery history:", data);
      setBatteryHistory(data);
    })
    .catch((err) => {
  console.error("Battery history error:", err);
  setBatteryHistory(null);
})
}, [selectedBattery]);
const historyCycles = batteryHistory?.available_cycles || [];

  // -----------------------------
  // Battery selector handler
  // -----------------------------

  const handleBatteryChange = (event) => {
    setSelectedBattery(event.target.value);
  };

  // -----------------------------
  // Loading state
  // -----------------------------

  if (loading) {
    return (
      <div className="app">
        <header className="header">
          <div>
            <h1>BATTERYIQ</h1>
            <p>AI-POWERED PREDICTIVE BATTERY HEALTH ANALYZER</p>
          </div>
        </header>

        <main className="dashboard">
          <div className="loading-state">
            <h2>Loading BatteryIQ...</h2>
            <p>Connecting to the BatteryIQ backend.</p>
          </div>
        </main>
      </div>
    );
  }

  // -----------------------------
  // Main dashboard
  // -----------------------------

  return (
    <div className="app">

      {/* =========================
          HEADER
      ========================== */}

      <header className="header">
        <div className="header-brand">
          <h1>BATTERYIQ</h1>
          <p>AI-POWERED PREDICTIVE BATTERY HEALTH ANALYZER</p>
        </div>

        <div className="battery-selector">
          <label htmlFor="battery-select">
            SELECT BATTERY
          </label>

          {loading ? (
  <div className="selector-status">
    Loading batteries...
  </div>
) : error ? (
  <div className="selector-status error">
    Unable to load batteries.
  </div>
) : (
  <select
    value={selectedBattery}
    onChange={(e) => setSelectedBattery(e.target.value)}
  >
    {batteries.map((battery) => (
      <option
        key={battery.battery_id}
        value={battery.battery_id}
      >
        {battery.battery_id}
      </option>
    ))}
  </select>
)}
        </div>
      </header>

      {/* =========================
          ERROR MESSAGE
      ========================== */}

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <main className="dashboard">

        {/* =========================
            BATTERY OVERVIEW
        ========================== */}

        <section className="overview-section">

          <div className="section-heading">
            <div>
              <h2>Battery Health Overview</h2>

              {selectedBattery && (
                <p>
                  Battery: <strong>{selectedBattery}</strong>
                </p>
              )}
            </div>
          </div>

          {/* =========================
              KPI CARDS
          ========================== */}

<div className="kpi-grid">
  <KpiCard
    label="SOH RANGE"
    value={
      batteryInfo?.soh_range
        ? `${batteryInfo.soh_range.min.toFixed(1)}% – ${batteryInfo.soh_range.max.toFixed(1)}%`
        : "--"
    }
    description="Observed SOH range for the selected battery"
  />

  <KpiCard
    label="TOTAL CYCLES"
    value={batteryInfo?.total_cycles ?? "--"}
    description="Available discharge cycles"
  />

  <KpiCard
    label="RUL STATUS"
    value={
      batteryInfo?.is_rul_valid
        ? "Available"
        : "Unavailable"
    }
    description={
      batteryInfo?.is_rul_valid
        ? "RUL prediction is available"
        : "Battery is right-censored"
    }
  />
</div>
        </section>

        {/* =========================
            BATTERY INFORMATION
        ========================== */}

        <section className="section">

          <div className="section-heading">
            <div>
              <h2>Battery Information</h2>
              <p>
                Information provided by the BatteryIQ backend.
              </p>
            </div>
          </div>

          {infoLoading ? (
            <div className="loading-state small">
              Loading battery information...
            </div>
          ) : batteryInfo ? (
            <div className="info-grid">

              <div className="info-card">
                <span className="info-label">
                  BATTERY ID
                </span>

                <span className="info-value">
                  {batteryInfo.battery_id}
                </span>
              </div>

              <div className="info-card">
                <span className="info-label">
                  TOTAL CYCLES
                </span>

                <span className="info-value">
                  {batteryInfo.total_cycles}
                </span>
              </div>

              <div className="info-card">
                <span className="info-label">
                  SOH RANGE
                </span>

                <span className="info-value">
                  {batteryInfo.soh_range
                    ? `${batteryInfo.soh_range.min.toFixed(1)}% – ${batteryInfo.soh_range.max.toFixed(1)}%`
                    : "—"}
                </span>
              </div>

              <div className="info-card">
                <span className="info-label">
                  RUL STATUS
                </span>

                <span className="info-value">
                  {batteryInfo.is_rul_valid
                    ? "Available"
                    : "Unavailable"}
                </span>
              </div>

            </div>

          ) : (
            <div className="empty-state">
              No battery information available.
            </div>
          )}

        </section>

        {/* =========================
            HEALTH HISTORY
        ========================== */}
        <section className="section">

  <div className="section-heading">
    <div>
      <h2>Battery Health History</h2>

      <p>
        Historical cycle-wise data will be displayed here
        once the dashboard-ready historical API is available.
      </p>
    </div>
  </div>

  <div className="chart-section">
  <div className="section-note">
    {batteryHistory
      ? `${historyCycles.length} cycles available for ${selectedBattery}`
      : "Historical cycle data is loading..."}
  </div>

  <BatteryChart
  title="SOH / Capacity Degradation"
  data={[]}
  dataKey="soh"
  yAxisLabel="SOH (%)"
  lineName="SOH"
  emptyMessage="Cycle-wise SOH data is not available yet."
/>
</div>

</section>

<section className="section">

  <div className="section-heading">
    <div>
      <h2>Battery Condition</h2>

      <p>
        Classification status from the BatteryIQ backend.
      </p>
    </div>
  </div>

  <ConditionStatus
  condition={condition}
  status={
    conditionLoading
      ? "loading"
      : conditionStatus
  }
/>

</section>

        {/* =========================
            MEASUREMENTS
        ========================== */}

        <section className="section">

  <div className="section-heading">
    <div>
      <h2>Battery Measurements</h2>

      <p>
        Sensor measurements across battery cycles.
      </p>
    </div>
  </div>

  <div className="charts-grid">

    <BatteryChart
  title="Voltage Curve"
  data={[]}
  dataKey="voltage"
  yAxisLabel="Voltage (V)"
  lineName="Voltage"
  emptyMessage="Voltage curve data is not available yet."
/>

    <BatteryChart
  title="Current Curve"
  data={[]}
  dataKey="current"
  yAxisLabel="Current (A)"
  lineName="Current"
  emptyMessage="Current curve data is not available yet."
/>

    <BatteryChart
  title="Temperature Curve"
  data={[]}
  dataKey="temperature"
  yAxisLabel="Temperature (°C)"
  lineName="Temperature"
  emptyMessage="Temperature curve data is not available yet."
/>

  </div>

</section>

        {/* =========================
            RUL
        ========================== */}

        <section className="section">

  <div className="section-heading">
    <div>
      <h2>Predictions</h2>

      <p>
        Predictions from the approved BatteryIQ models.
      </p>
    </div>
  </div>

  <div className="prediction-grid">

  <SohPrediction
    prediction={sohPrediction}
    loading={sohLoading}
    error={sohError}
  />

  <RulPrediction
    batteryId={selectedBattery}
    rulValid={batteryInfo?.is_rul_valid ?? false}
    prediction={rulPrediction}
    loading={rulLoading}
    error={rulError}
  />

</div>

</section>

        {/* =========================
            FEATURE CONTRIBUTION
        ========================== */}

        <section className="section">

  <div className="section-heading">
    <div>
      <h2>Feature Contribution</h2>

      <p>
        Model feature contributions generated by the
        approved SHAP implementation.
      </p>
    </div>
  </div>

<ShapChart
  featureContributions={featureContributions}
  loading={shapLoading}
  error={shapError}
/>

</section>

      </main>

      {/* =========================
          FOOTER
      ========================== */}

      <footer className="footer">
        <p>
          BatteryIQ — AI-Powered Predictive Battery Health Analyzer
        </p>
      </footer>

    </div>
  );
}

export default App;