import { MapPin, Tag } from 'lucide-react'
import ResultsPanel from './ResultsPanel'

function parseMarkdown(text) {
  // Bold
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  // Newlines
  text = text.replace(/\n/g, '<br/>')
  return text
}

const INTERRUPT_ICONS = {
  location: MapPin,
  category: Tag,
}

export default function ChatMessage({ message }) {
  const isUser = message.role === 'user'
  const isInterrupt = message.requiresInput
  const InterruptIcon = INTERRUPT_ICONS[message.inputPrompt] || MapPin

  if (isUser) {
    return (
      <div className="animate-fade-slide-up" style={{
        display: 'flex',
        justifyContent: 'flex-end',
        padding: '2px 0',
      }}>
        <div style={{
          background: 'linear-gradient(135deg, #6366f1, #4f46e5)',
          color: 'white',
          borderRadius: '14px 4px 14px 14px',
          padding: '10px 14px',
          maxWidth: '75%',
          fontSize: 14,
          lineHeight: 1.55,
          boxShadow: '0 2px 12px rgba(99,102,241,0.25)',
        }}>
          {message.content}
        </div>
      </div>
    )
  }

  // Assistant message
  return (
    <div className="animate-fade-slide-up" style={{
      display: 'flex',
      alignItems: 'flex-start',
      gap: 10,
      padding: '2px 0',
    }}>
      {/* Bot avatar */}
      <div style={{
        width: 34,
        height: 34,
        borderRadius: 10,
        background: 'linear-gradient(135deg, #6366f1, #22d3ee)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        fontSize: 15,
        boxShadow: '0 0 14px rgba(99,102,241,0.25)',
      }}>
        🤖
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Text bubble */}
        {message.content && (
          <div style={{
            background: isInterrupt
              ? 'linear-gradient(135deg, rgba(245,158,11,0.1), rgba(245,158,11,0.05))'
              : 'var(--bg-card)',
            border: isInterrupt
              ? '1px solid rgba(245,158,11,0.3)'
              : '1px solid var(--border)',
            borderRadius: '4px 14px 14px 14px',
            padding: '12px 14px',
            fontSize: 14,
            color: 'var(--text-secondary)',
            lineHeight: 1.65,
            marginBottom: message.data ? 10 : 0,
          }}>
            {isInterrupt && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                marginBottom: 8,
                color: '#f59e0b',
                fontSize: 12,
                fontFamily: 'JetBrains Mono, monospace',
                letterSpacing: '0.04em',
              }}>
                <InterruptIcon size={12} />
                NEEDS YOUR INPUT
              </div>
            )}
            <span dangerouslySetInnerHTML={{ __html: parseMarkdown(message.content) }} />
          </div>
        )}

        {/* Results panel (only for complete state) */}
        {message.data && message.agentState === 'complete' && (
          <ResultsPanel data={message.data} />
        )}

        {/* Timestamp */}
        <div style={{
          fontSize: 10,
          color: 'var(--text-muted)',
          marginTop: 4,
          fontFamily: 'JetBrains Mono, monospace',
        }}>
          {message.timestamp?.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          {message.agentState && message.agentState !== 'idle' && (
            <span style={{ marginLeft: 6, opacity: 0.6 }}>· {message.agentState.replace(/_/g, ' ')}</span>
          )}
        </div>
      </div>
    </div>
  )
}
