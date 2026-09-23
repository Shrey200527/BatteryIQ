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

  const data = await response.json();

  console.log("Battery API response:", data);

  return data;
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

  const data = await response.json();

  console.log("Battery info response:", data);

  return data;
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
      body: JSON.stringify({
        soh,
      }),
    }
  );

  if (!response.ok) {
    throw new Error("Failed to fetch battery condition.");
  }

  const data = await response.json();

  console.log("Condition API response:", data);

  return data;
}

/**
 * Get SHAP feature contributions for a battery cycle.
 */
export async function getShapExplanation(batteryId, cycle, features) {
  const response = await fetch(
    `${API_BASE_URL}/api/predict/explain`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        battery_id: batteryId,
        cycle: cycle,
        ...features,
      }),
    }
  );

  if (!response.ok) {
    throw new Error("Failed to fetch SHAP explanation.");
  }

  const data = await response.json();

  console.log("SHAP API response:", data);

  return data;
}

/**
 * Get historical information for a battery.
 */
export async function getBatteryHistory(batteryId) {
  const response = await fetch(
    `${API_BASE_URL}/api/batteries/${batteryId}/history`
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch history for ${batteryId}.`);
  }

  const data = await response.json();

  console.log("Battery history response:", data);

  return data;
}

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
        cycle: cycle,
        ...features,
      }),
    }
  );

  if (!response.ok) {
    throw new Error("Failed to fetch SOH prediction.");
  }

  const data = await response.json();

  console.log("SOH prediction response:", data);

  return data;
}