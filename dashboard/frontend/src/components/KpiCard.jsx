function KpiCard({ label, value, description }) {
  return (
    <div className="kpi-card">
      <span className="kpi-label">
        {label}
      </span>

      <div className="kpi-value">
        {value}
      </div>

      <span className="kpi-description">
        {description}
      </span>
    </div>
  );
}

export default KpiCard;