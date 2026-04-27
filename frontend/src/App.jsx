import { useRef, useEffect } from 'react'
import { useChat } from './hooks/useChat'
import Header from './components/Header'
import AgentStatusBar from './components/AgentStatusBar'
import ChatMessage from './components/ChatMessage'
import ChatInput from './components/ChatInput'
import LoadingThinking from './components/LoadingThinking'
import WelcomeScreen from './components/WelcomeScreen'

export default function App() {
  const {
    messages,
    isLoading,
    agentState,
    sendMessage,
    resetSession,
    sessionId,
  } = useChat()

  const bottomRef = useRef(null)
  const messagesRef = useRef(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const hasMessages = messages.length > 0

  return (
    <div style={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      background: 'var(--bg-deep)',
      overflow: 'hidden',
    }}>
      {/* Ambient background glow */}
      <div style={{
        position: 'fixed',
        top: '-20%',
        left: '50%',
        transform: 'translateX(-50%)',
        width: '60vw',
        height: '40vh',
        background: 'radial-gradient(ellipse at center, rgba(99,102,241,0.08) 0%, transparent 70%)',
        pointerEvents: 'none',
        zIndex: 0,
      }} />

      {/* Header */}
      <Header onReset={resetSession} sessionId={sessionId} />

      {/* Agent status bar */}
      <AgentStatusBar agentState={agentState} />

      {/* Main content area */}
      <div
        ref={messagesRef}
        style={{
          flex: 1,
          overflowY: 'auto',
          position: 'relative',
          zIndex: 1,
        }}
      >
        {!hasMessages ? (
          <WelcomeScreen onExample={sendMessage} />
        ) : (
          <div style={{
            maxWidth: 760,
            margin: '0 auto',
            padding: '24px 20px',
            display: 'flex',
            flexDirection: 'column',
            gap: 14,
          }}>
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} />
            ))}

            {/* Loading indicator */}
            {isLoading && <LoadingThinking agentState={agentState} />}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div style={{
        background: 'var(--bg-card)',
        borderTop: '1px solid var(--border)',
        padding: '16px 20px',
        position: 'relative',
        zIndex: 2,
        flexShrink: 0,
      }}>
        <div style={{ maxWidth: 760, margin: '0 auto' }}>
          <ChatInput
            onSend={sendMessage}
            isLoading={isLoading}
            showSuggestions={!hasMessages}
            placeholder={
              agentState === 'waiting_for_location'
                ? '📍 Enter a city or area (e.g., "New York", "downtown Chicago")…'
                : agentState === 'waiting_for_category'
                ? '🏪 Enter a business type (e.g., "Gyms", "Hotels", "Restaurants")…'
                : 'Ask about any business — e.g. "Best coffee shops in Chicago"'
            }
          />
          <p style={{
            fontSize: 11,
            color: 'var(--text-muted)',
            textAlign: 'center',
            marginTop: 8,
            fontFamily: 'JetBrains Mono, monospace',
            letterSpacing: '0.03em',
          }}>
            Press <kbd style={{
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              padding: '1px 5px',
              fontSize: 10,
            }}>Enter</kbd> to send · <kbd style={{
              background: 'var(--bg-elevated)',
              border: '1px solid var(--border)',
              borderRadius: 4,
              padding: '1px 5px',
              fontSize: 10,
            }}>Shift+Enter</kbd> for newline
          </p>
        </div>
      </div>
    </div>
  )
}
