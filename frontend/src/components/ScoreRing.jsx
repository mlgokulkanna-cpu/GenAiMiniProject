export default function ScoreRing({ score, size = 72 }) {
  const radius = (size - 10) / 2
  const circumference = 2 * Math.PI * radius
  const filled = (score / 10) * circumference
  const color = score >= 7.5 ? '#10b981' : score >= 5 ? '#f59e0b' : '#f43f5e'
  const glow = score >= 7.5
    ? '0 0 16px rgba(16,185,129,0.4)'
    : score >= 5
    ? '0 0 16px rgba(245,158,11,0.4)'
    : '0 0 16px rgba(244,63,94,0.4)'

  return (
    <div style={{ position: 'relative', width: size, height: size, flexShrink: 0 }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        {/* Track */}
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none"
          stroke="var(--bg-elevated)"
          strokeWidth={5}
        />
        {/* Progress */}
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none"
          stroke={color}
          strokeWidth={5}
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference}`}
          style={{
            filter: `drop-shadow(${glow})`,
            transition: 'stroke-dasharray 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)',
          }}
        />
      </svg>
      {/* Center label */}
      <div style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        color,
        lineHeight: 1,
      }}>
        <span style={{ fontFamily: 'Syne, sans-serif', fontWeight: 700, fontSize: size * 0.25 }}>
          {score.toFixed(1)}
        </span>
        <span style={{ fontSize: size * 0.14, color: 'var(--text-muted)', marginTop: 1 }}>/ 10</span>
      </div>
    </div>
  )
}
