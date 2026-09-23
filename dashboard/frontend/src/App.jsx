import { useEffect, useState } from "react";
import "./App.css";

import {
  getBatteries,
  getBatteryInfo,
  getBatteryHistory,
  getLatestFeatures,
  getSohPrediction,
  getRulPrediction,
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
  // Battery state
  // -----------------------------

  const [batteries, setBatteries] = useState([]);
  const [selectedBattery, setSelectedBattery] = useState("");

  const [batteryInfo, setBatteryInfo] = useState(null);
  const [batteryHistory, setBatteryHistory] = useState(null);
  const [latestFeatures, setLatestFeatures] = useState(null);

  // -----------------------------
  // Loading / error state
  // -----------------------------

  const [loading, setLoading] = useState(true);

  const [error, setError] = useState("");

  // -----------------------------
  // SOH prediction
  // -----------------------------

  const [sohPrediction, setSohPrediction] = useState(null);
  const [sohError, setSohError] = useState("");

  // -----------------------------
  // RUL prediction
  // -----------------------------

  const [rulPrediction, setRulPrediction] = useState(null);
  const [rulError, setRulError] = useState("");

  // -----------------------------
  // SHAP explanation
  // -----------------------------

  const [featureContributions, setFeatureContributions] = useState([]);
  const [shapReady, setShapReady] = useState(false);
  const [shapError, setShapError] = useState("");

  // -----------------------------
  // Condition classification
  // -----------------------------
  //
  // Numerical condition thresholds are intentionally
  // pending approval in the ML contract.
  //

  const [condition] = useState(null);
  const [conditionStatus] = useState("thresholds_pending");
  const [conditionLoading] = useState(false);

  // Loading state is derived from the selected battery and request results.
  // This avoids synchronous state updates inside effects.
  const infoLoading = Boolean(
    selectedBattery && !batteryInfo && !error
  );
  const featuresLoading = Boolean(
    selectedBattery && !latestFeatures && !error
  );
  const sohLoading = Boolean(
    selectedBattery &&
      latestFeatures &&
      batteryInfo &&
      !sohPrediction &&
      !sohError
  );
  const rulLoading = Boolean(
    selectedBattery &&
      latestFeatures &&
      batteryInfo?.is_rul_valid &&
      !rulPrediction &&
      !rulError
  );
  const shapLoading = Boolean(
    selectedBattery &&
      latestFeatures &&
      batteryInfo &&
      !shapReady &&
      !shapError
  );

  const handleBatteryChange = (event) => {
    const batteryId = event.target.value;

    setSelectedBattery(batteryId);
    setError("");

    setBatteryInfo(null);
    setBatteryHistory(null);
    setLatestFeatures(null);

    setSohPrediction(null);
    setSohError("");

    setRulPrediction(null);
    setRulError("");

    setFeatureContributions([]);
    setShapReady(false);
    setShapError("");
  };

  // ============================================================
  // Load available batteries
  // ============================================================

  useEffect(() => {
    getBatteries()
      .then((data) => {
        console.log("Battery list received:", data);

        const availableBatteries = data.batteries || [];

        setBatteries(availableBatteries);

        // Automatically select the first battery.
        if (availableBatteries.length > 0) {
          setSelectedBattery(
            availableBatteries[0].battery_id
          );
        }
      })
      .catch((err) => {
        console.error("Battery list error:", err);
        setError(
          "Unable to connect to BatteryIQ backend."
        );
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  // ============================================================
  // Load selected battery information, history and features
  // ============================================================

  useEffect(() => {
    if (!selectedBattery) {
      return;
    }

    let cancelled = false;

    Promise.all([
      getBatteryInfo(selectedBattery),
      getBatteryHistory(selectedBattery),
      getLatestFeatures(selectedBattery),
    ])
      .then(([info, history, featureData]) => {
        if (cancelled) {
          return;
        }

        console.log(
          "Selected battery information:",
          info
        );

        console.log(
          "Selected battery history:",
          history
        );

        console.log(
          "Latest ML features:",
          featureData
        );

        setBatteryInfo(info);
        setBatteryHistory(history);
        setLatestFeatures(featureData);
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }

        console.error(
          "Selected battery data error:",
          err
        );

        setBatteryInfo(null);
        setBatteryHistory(null);
        setLatestFeatures(null);

        setError(
          "Unable to load selected battery data."
        );
      });

    return () => {
      cancelled = true;
    };
  }, [selectedBattery]);

  // ============================================================
  // Run ML predictions and SHAP explanation
  // ============================================================

  useEffect(() => {
    if (
      !selectedBattery ||
      !latestFeatures?.features ||
      latestFeatures.cycle == null ||
      !batteryInfo
    ) {
      return;
    }

    const cycle = latestFeatures.cycle;
    const features = { ...latestFeatures.features };
    delete features.battery_id;

    let cancelled = false;

    const runPredictions = async () => {
      const rulPromise = batteryInfo.is_rul_valid
        ? getRulPrediction(
            selectedBattery,
            "xgboost",
            features
          )
        : Promise.resolve(null);

      const [sohResult, rulResult, shapResult] =
        await Promise.allSettled([
          getSohPrediction(
            selectedBattery,
            cycle,
            features
          ),
          rulPromise,
          getShapExplanation(
            selectedBattery,
            cycle,
            features
          ),
        ]);

      if (cancelled) {
        return;
      }

      // --------------------------------------------------------
      // SOH prediction
      // --------------------------------------------------------

      if (sohResult.status === "fulfilled") {
        console.log(
          "SOH prediction:",
          sohResult.value
        );

        setSohPrediction(sohResult.value);
        setSohError("");
      } else {
        console.error(
          "SOH prediction error:",
          sohResult.reason
        );

        setSohPrediction(null);
        setSohError(
          sohResult.reason?.message ||
            "SOH prediction failed."
        );
      }

      // --------------------------------------------------------
      // RUL prediction
      // --------------------------------------------------------

      if (!batteryInfo.is_rul_valid) {
        /*
         * Right-censored battery:
         * no artificial RUL prediction.
         */
        setRulPrediction(null);
        setRulError("");
      } else if (rulResult.status === "fulfilled") {
        console.log(
          "RUL prediction:",
          rulResult.value
        );

        setRulPrediction(rulResult.value);
        setRulError("");
      } else {
        console.error(
          "RUL prediction error:",
          rulResult.reason
        );

        setRulPrediction(null);
        setRulError(
          rulResult.reason?.message ||
            "RUL prediction failed."
        );
      }

      // --------------------------------------------------------
      // SHAP explanation
      // --------------------------------------------------------

      if (shapResult.status === "fulfilled") {
        console.log(
          "SHAP explanation:",
          shapResult.value
        );

        /*
         * Backend returns:
         *
         * feature_contributions: {
         *   feature_name: {
         *     shap_value: ...,
         *     feature_value: ...
         *   }
         * }
         *
         * ShapChart expects an array.
         */

        const contributions = Object.entries(
          shapResult.value.feature_contributions || {}
        )
          .map(([feature, values]) => ({
            feature,
            shap_value: Number(
              values?.shap_value ?? 0
            ),
            feature_value: Number(
              values?.feature_value ?? 0
            ),
          }))
          .sort(
            (a, b) =>
              Math.abs(b.shap_value) -
              Math.abs(a.shap_value)
          );

        setFeatureContributions(contributions);
        setShapError("");
        setShapReady(true);
      } else {
        console.error(
          "SHAP explanation error:",
          shapResult.reason
        );

        setFeatureContributions([]);
        setShapError(
          shapResult.reason?.message ||
            "SHAP explanation failed."
        );
        setShapReady(true);
      }
    };

    runPredictions().catch((err) => {
      if (cancelled) {
        return;
      }

      console.error(
        "Unexpected prediction error:",
        err
      );

      setSohPrediction(null);
      setSohError(
        err.message ||
          "Prediction request failed."
      );

      if (batteryInfo.is_rul_valid) {
        setRulPrediction(null);
        setRulError(
          err.message ||
            "RUL prediction failed."
        );
      }

      setFeatureContributions([]);
      setShapError(
        err.message ||
          "SHAP explanation failed."
      );
      setShapReady(true);
    });

    return () => {
      cancelled = true;
    };
  }, [
    selectedBattery,
    latestFeatures,
    batteryInfo,
  ]);

  // ============================================================
  // Derived dashboard data
  // ============================================================

  const historyCycles =
    batteryHistory?.available_cycles || [];

  const historyData =
    batteryHistory?.history || [];

  // ============================================================
  // Loading state
  // ============================================================

  if (loading) {
    return (
      <div className="app">
        <header className="header">
          <div>
            <h1>BATTERYIQ</h1>
            <p>
              AI-POWERED PREDICTIVE BATTERY HEALTH ANALYZER
            </p>
          </div>
        </header>

        <main className="dashboard">
          <div className="loading-state">
            <h2>Loading BatteryIQ...</h2>
            <p>
              Connecting to the BatteryIQ backend.
            </p>
          </div>
        </main>
      </div>
    );
  }

  // ============================================================
  // Main dashboard
  // ============================================================

  return (
    <div className="app">

      {/* ========================================================
          HEADER
      ========================================================= */}

      <header className="header">
        <div className="header-brand">
          <h1>BATTERYIQ</h1>

          <p>
            AI-POWERED PREDICTIVE BATTERY HEALTH ANALYZER
          </p>
        </div>

        <div className="battery-selector">
          <label htmlFor="battery-select">
            SELECT BATTERY
          </label>

          {error ? (
            <div className="selector-status error">
              Unable to load batteries.
            </div>
          ) : (
            <select
              id="battery-select"
              value={selectedBattery}
              onChange={handleBatteryChange}
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

      {/* ========================================================
          GLOBAL ERROR
      ========================================================= */}

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <main className="dashboard">

        {/* ======================================================
            BATTERY OVERVIEW
        ====================================================== */}

        <section className="overview-section">

          <div className="section-heading">
            <div>
              <h2>Battery Health Overview</h2>

              {selectedBattery && (
                <p>
                  Battery:{" "}
                  <strong>
                    {selectedBattery}
                  </strong>
                </p>
              )}
            </div>
          </div>

          <div className="kpi-grid">

            <KpiCard
              label="SOH RANGE"
              value={
                batteryInfo?.soh_range
                  ? `${batteryInfo.soh_range.min.toFixed(
                      1
                    )}% – ${batteryInfo.soh_range.max.toFixed(
                      1
                    )}%`
                  : "--"
              }
              description="Observed SOH range for the selected battery"
            />

            <KpiCard
              label="TOTAL CYCLES"
              value={
                batteryInfo?.total_cycles ??
                "--"
              }
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

        {/* ======================================================
            BATTERY INFORMATION
        ====================================================== */}

        <section className="section">

          <div className="section-heading">
            <div>
              <h2>Battery Information</h2>

              <p>
                Information provided by the
                BatteryIQ backend.
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
                    ? `${batteryInfo.soh_range.min.toFixed(
                        1
                      )}% – ${batteryInfo.soh_range.max.toFixed(
                        1
                      )}%`
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
                    : "Right-censored"}
                </span>
              </div>

            </div>

          ) : (

            <div className="empty-state">
              No battery information available.
            </div>

          )}

        </section>

        {/* ======================================================
            HEALTH HISTORY
        ====================================================== */}

        <section className="section">

          <div className="section-heading">
            <div>
              <h2>Battery Health History</h2>

              <p>
                Cycle-wise battery health data
                from the validated dataset.
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
              title="SOH Degradation"
              data={historyData}
              dataKey="soh"
              yAxisLabel="SOH (%)"
              lineName="SOH"
              emptyMessage="Cycle-wise SOH data is not available yet."
            />

          </div>

        </section>

        {/* ======================================================
            BATTERY CONDITION
        ====================================================== */}

        <section className="section">

          <div className="section-heading">
            <div>
              <h2>Battery Condition</h2>

              <p>
                Condition classification remains
                pending until numerical thresholds
                are approved in the ML contract.
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

        {/* ======================================================
            MEASUREMENTS
        ====================================================== */}

        <section className="section">

          <div className="section-heading">
            <div>
              <h2>Battery Measurements</h2>

              <p>
                Cycle-level voltage, current and
                temperature statistics.
              </p>
            </div>
          </div>

          <div className="charts-grid">

            <BatteryChart
              title="Voltage Curve"
              data={historyData}
              dataKey="voltage_mean"
              yAxisLabel="Voltage (V)"
              lineName="Voltage Mean"
              emptyMessage="Voltage data is not available yet."
            />

            <BatteryChart
              title="Current Curve"
              data={historyData}
              dataKey="current_mean"
              yAxisLabel="Current (A)"
              lineName="Current Mean"
              emptyMessage="Current data is not available yet."
            />

            <BatteryChart
              title="Temperature Curve"
              data={historyData}
              dataKey="temperature_mean"
              yAxisLabel="Temperature (°C)"
              lineName="Temperature Mean"
              emptyMessage="Temperature data is not available yet."
            />

          </div>

        </section>

        {/* ======================================================
            PREDICTIONS
        ====================================================== */}

        <section className="section">

          <div className="section-heading">
            <div>
              <h2>Predictions</h2>

              <p>
                Predictions from the approved
                BatteryIQ inference models.
              </p>
            </div>
          </div>

          {featuresLoading ? (
            <div className="loading-state small">
              Preparing validated ML features...
            </div>
          ) : (

            <div className="prediction-grid">

              <SohPrediction
                prediction={sohPrediction}
                loading={sohLoading}
                error={sohError}
              />

              <RulPrediction
                batteryId={selectedBattery}
                rulValid={
                  batteryInfo?.is_rul_valid ??
                  false
                }
                prediction={rulPrediction}
                loading={rulLoading}
                error={rulError}
              />

            </div>

          )}

        </section>

        {/* ======================================================
            SHAP FEATURE CONTRIBUTION
        ====================================================== */}

        <section className="section">

          <div className="section-heading">
            <div>
              <h2>Feature Contribution</h2>

              <p>
                Model feature contributions generated
                by the approved SHAP implementation.
              </p>
            </div>
          </div>

          <ShapChart
            featureContributions={
              featureContributions
            }
            loading={shapLoading}
            error={shapError}
          />

        </section>

      </main>

      {/* ========================================================
          FOOTER
      ========================================================= */}

      <footer className="footer">
        <p>
          BatteryIQ — AI-Powered Predictive Battery
          Health Analyzer
        </p>
      </footer>

    </div>
  );
}

export default App;