import BusinessCard from './BusinessCard'

export default function ResultsPanel({ data }) {
  if (!data) return null

  if (data.type === 'single_business') {
    return (
      <div className="animate-fade-slide-up" style={{ marginTop: 8 }}>
        <BusinessCard data={data.data} />
      </div>
    )
  }

  if (data.type === 'top5') {
    const { businesses = [], winner, winner_reason, category, location } = data

    return (
      <div className="animate-fade-slide-up" style={{ marginTop: 8 }}>
        {/* Header */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(99,102,241,0.12), rgba(34,211,238,0.08))',
          border: '1px solid rgba(99,102,241,0.25)',
          borderRadius: 14,
          padding: '16px 18px',
          marginBottom: 12,
        }}>
          <div style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace', color: 'var(--text-muted)', marginBottom: 4, letterSpacing: '0.06em' }}>
            TOP {businesses.length} {(category || 'BUSINESSES').toUpperCase()} · {(location || '').toUpperCase()}
          </div>
          <div style={{ fontFamily: 'Syne, sans-serif', fontWeight: 700, fontSize: 15, color: 'var(--text-primary)', marginBottom: 4 }}>
            🏆 Winner: {winner}
          </div>
          <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            {winner_reason}
          </div>
        </div>

        {/* Business list */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {businesses.map((biz, i) => (
            <div
              key={biz.business_name}
              className={`delay-${(i + 1) * 100}`}
              style={{ animation: 'fadeSlideUp 0.4s cubic-bezier(0.22, 1, 0.36, 1) both', animationDelay: `${i * 80}ms` }}
            >
              <BusinessCard
                data={biz}
                rank={i + 1}
                isWinner={biz.business_name === winner}
              />
            </div>
          ))}
        </div>
      </div>
    )
  }

  return null
}
