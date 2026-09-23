import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

function BatteryChart({
  title,
  data = [],
  dataKey,
  xAxisKey = "cycle",
  yAxisLabel = "",
  lineName = "",
  emptyMessage = "Historical data not available yet.",
}) {
  return (
    <div className="chart-card">
      <div className="chart-title">{title}</div>

      {data.length === 0 ? (
        <div className="chart-placeholder compact">
          <span>{emptyMessage}</span>
        </div>
      ) : (
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={data}
              margin={{
                top: 10,
                right: 20,
                left: 10,
                bottom: 10,
              }}
            >
              <CartesianGrid strokeDasharray="3 3" />

              <XAxis
                dataKey={xAxisKey}
                label={{
                  value: "Cycle",
                  position: "insideBottom",
                  offset: -5,
                }}
              />

              <YAxis
                label={{
                  value: yAxisLabel,
                  angle: -90,
                  position: "insideLeft",
                }}
              />

              <Tooltip />

              <Line
                type="monotone"
                dataKey={dataKey}
                name={lineName || dataKey}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export default BatteryChart;