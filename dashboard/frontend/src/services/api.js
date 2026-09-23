const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * Fetch all available batteries.
 */
export async function getBatteries() {
  const response = await fetch(`${API_BASE_URL}/api/batteries`);

  if (!response.ok) {
    throw new Error("Failed to fetch batteries.");
  }

  return response.json();
}

/**
 * Fetch information about a specific battery.
 */
export async function getBatteryInfo(batteryId) {
  const response = await fetch(
    `${API_BASE_URL}/api/batteries/${batteryId}`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch information for ${batteryId}.`);
  }

  return response.json();
}

/**
 * Fetch dashboard-ready historical data.
 */
export async function getBatteryHistory(batteryId) {
  const response = await fetch(
    `${API_BASE_URL}/api/batteries/${batteryId}/history`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch history for ${batteryId}.`);
  }

  return response.json();
}

/**
 * Fetch the latest complete ML feature vector.
 */
export async function getLatestFeatures(batteryId) {
  const response = await fetch(
    `${API_BASE_URL}/api/batteries/${batteryId}/latest-features`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch latest features for ${batteryId}.`);
  }

  return response.json();
}

/**
 * Predict SOH for a battery cycle.
 */
export async function getSohPrediction(batteryId, cycle, features) {
  const response = await fetch(
    `${API_BASE_URL}/api/predict/soh`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        battery_id: batteryId,
        cycle,
        features,
      }),
    }
  );

  if (!response.ok) {
    throw new Error("Failed to fetch SOH prediction.");
  }

  return response.json();
}

/**
 * Predict RUL using XGBoost.
 *
 * XGBoost uses a single cycle's feature vector.
 */
export async function getRulPrediction(
  batteryId,
  model,
  features
) {
  const response = await fetch(
    `${API_BASE_URL}/api/predict/rul`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        battery_id: batteryId,
        model,
        features,
      }),
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new Error(
      errorData?.detail?.message ||
        "Failed to fetch RUL prediction."
    );
  }

  return response.json();
}

/**
 * Get battery condition from SOH.
 */
export async function getBatteryCondition(soh) {
  const response = await fetch(
    `${API_BASE_URL}/api/predict/condition`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ soh }),
    }
  );

  if (!response.ok) {
    throw new Error("Failed to fetch battery condition.");
  }

  return response.json();
}

/**
 * Get SHAP feature contributions.
 */
export async function getShapExplanation(
  batteryId,
  cycle,
  features
) {
  const response = await fetch(
    `${API_BASE_URL}/api/predict/explain`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        battery_id: batteryId,
        cycle,
        features,
      }),
    }
  );

  if (!response.ok) {
    throw new Error("Failed to fetch SHAP explanation.");
  }

  return response.json();
}