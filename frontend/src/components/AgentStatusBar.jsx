import React from 'react'
import { Search, Brain, ShieldCheck, CheckCircle2, AlertCircle, Loader2, Zap } from 'lucide-react'

const STATES = {
  idle:                   { label: 'Ready',           color: '#44445a', icon: Zap,          bg: 'rgba(68,68,90,0.15)' },
  triaging:               { label: 'Triaging',        color: '#a5b4fc', icon: Loader2,       bg: 'rgba(165,180,252,0.12)', spin: true },
  waiting_for_location:   { label: 'Needs Location',  color: '#f59e0b', icon: AlertCircle,   bg: 'rgba(245,158,11,0.12)' },
  waiting_for_category:   { label: 'Needs Category',  color: '#f59e0b', icon: AlertCircle,   bg: 'rgba(245,158,11,0.12)' },
  searching:              { label: 'Searching',        color: '#22d3ee', icon: Search,        bg: 'rgba(34,211,238,0.12)', pulse: true },
  analyzing:              { label: 'Analyzing',        color: '#a5b4fc', icon: Brain,         bg: 'rgba(165,180,252,0.12)', pulse: true },
  verifying:              { label: 'Verifying',        color: '#10b981', icon: ShieldCheck,   bg: 'rgba(16,185,129,0.12)', pulse: true },
  complete:               { label: 'Complete',         color: '#10b981', icon: CheckCircle2,  bg: 'rgba(16,185,129,0.12)' },
  error:                  { label: 'Error',            color: '#f43f5e', icon: AlertCircle,   bg: 'rgba(244,63,94,0.12)' },
}

const PIPELINE = ['triaging', 'searching', 'analyzing', 'verifying', 'complete']

export default function AgentStatusBar({ agentState }) {
  const current = STATES[agentState] || STATES.idle
  const Icon = current.icon
  const currentIdx = PIPELINE.indexOf(agentState)

  return (
    <div style={{
      background: 'var(--bg-card)',
      borderBottom: '1px solid var(--border)',
      padding: '10px 20px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: 16,
    }}>
      {/* Current state badge */}
      <div className="agent-badge" style={{ background: current.bg, color: current.color }}>
        <Icon
          size={11}
          style={current.spin ? { animation: 'spin 1s linear infinite' } : undefined}
        />
        {current.pulse && (
          <span className="pulse-dot" style={{ background: current.color }} />
        )}
        {current.label}
      </div>

      {/* Pipeline progress */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        {PIPELINE.map((step, i) => {
          const s = STATES[step]
          const isDone = currentIdx > i
          const isActive = currentIdx === i
          return (
            <React.Fragment key={step}>
              <div
                title={s.label}
                style={{
                  width: isActive ? 24 : 8,
                  height: 6,
                  borderRadius: 99,
                  background: isDone
                    ? 'var(--accent-emerald)'
                    : isActive
                    ? current.color
                    : 'var(--bg-elevated)',
                  transition: 'all 0.3s ease',
                  opacity: isDone ? 0.8 : isActive ? 1 : 0.3,
                }}
              />
              {i < PIPELINE.length - 1 && (
                <div style={{
                  width: 12,
                  height: 1,
                  background: isDone ? 'var(--accent-emerald)' : 'var(--border)',
                  opacity: isDone ? 0.6 : 0.3,
                  transition: 'all 0.3s',
                }} />
              )}
            </React.Fragment>
          )
        })}
      </div>

      {/* Groq tag */}
      <div style={{
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: 10,
        color: 'var(--text-muted)',
        letterSpacing: '0.05em',
      }}>
        GROQ · SERPAPI · LANGGRAPH
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  )
}
