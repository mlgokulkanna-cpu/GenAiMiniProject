import { Search, Brain, ShieldCheck, Zap } from 'lucide-react'

const STEPS = [
  { state: 'triaging',  label: 'Analyzing your request…',       icon: Zap,         color: '#a5b4fc' },
  { state: 'searching', label: 'Fetching real-time data…',       icon: Search,      color: '#22d3ee' },
  { state: 'analyzing', label: 'Running sentiment analysis…',    icon: Brain,       color: '#a5b4fc' },
  { state: 'verifying', label: 'Verifying verdict quality…',     icon: ShieldCheck, color: '#10b981' },
]

export default function LoadingThinking({ agentState }) {
  const current = STEPS.find(s => s.state === agentState) || STEPS[0]
  const Icon = current.icon

  return (
    <div className="animate-fade-slide-up" style={{
      display: 'flex',
      alignItems: 'flex-start',
      gap: 12,
      padding: '4px 0',
    }}>
      {/* Avatar */}
      <div style={{
        width: 34,
        height: 34,
        borderRadius: 10,
        background: 'linear-gradient(135deg, #6366f1, #22d3ee)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        fontSize: 14,
        boxShadow: '0 0 16px rgba(99,102,241,0.3)',
      }}>
        🤖
      </div>

      {/* Thinking bubble */}
      <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: '4px 14px 14px 14px',
        padding: '12px 16px',
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
        minWidth: 200,
      }}>
        {/* Current step */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          color: current.color,
          fontFamily: 'DM Sans, sans-serif',
          fontSize: 13,
        }}>
          <Icon size={14} style={{ animation: 'spin 1.5s linear infinite', flexShrink: 0 }} />
          {current.label}
        </div>

        {/* Step dots */}
        <div style={{ display: 'flex', gap: 5 }}>
          {STEPS.map((s, i) => {
            const idx = STEPS.findIndex(x => x.state === agentState)
            const done = i < idx
            const active = i === idx
            return (
              <div key={s.state} style={{
                width: active ? 20 : 6,
                height: 6,
                borderRadius: 99,
                background: done ? '#10b981' : active ? s.color : 'var(--bg-elevated)',
                transition: 'all 0.3s ease',
                opacity: done ? 0.7 : active ? 1 : 0.3,
              }} />
            )
          })}
        </div>

        {/* Animated text dots */}
        <div style={{
          display: 'flex',
          gap: 4,
          alignItems: 'center',
        }}>
          {[0, 1, 2].map(i => (
            <div key={i} style={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              background: 'var(--text-muted)',
              animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
            }} />
          ))}
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40% { transform: translateY(-5px); opacity: 1; }
        }
      `}</style>
    </div>
  )
}
