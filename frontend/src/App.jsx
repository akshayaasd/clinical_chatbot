import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';
// Session ID generated using Math.random
// Let's use a simple random string for session id if uuid is not installed

const sessionId = Math.random().toString(36).substring(2, 15);

function App() {
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! Welcome to the Clinic. How can I help you today?", sender: 'bot' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { id: Date.now(), text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const botMessageId = Date.now() + 1;

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: userMessage.text
        }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let isFirstChunk = true;
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          setIsLoading(false);
          break;
        }
        
        buffer += decoder.decode(value, { stream: true });
        
        let newlineIndex;
        while ((newlineIndex = buffer.indexOf('\n\n')) >= 0) {
          const message = buffer.slice(0, newlineIndex);
          buffer = buffer.slice(newlineIndex + 2);
          
          if (message.startsWith('data: ')) {
            try {
              const data = JSON.parse(message.slice(6));
              if (data.content) {
                if (isFirstChunk) {
                  setIsLoading(false);
                  setMessages(prev => [...prev, { id: botMessageId, text: data.content, sender: 'bot' }]);
                  isFirstChunk = false;
                } else {
                  setMessages(prev => prev.map(msg => 
                    msg.id === botMessageId ? { ...msg, text: msg.text + data.content } : msg
                  ));
                }
              }
            } catch (err) {
              console.error("Error parsing stream JSON", err);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage = { 
        id: Date.now() + 2, 
        text: "We're currently experiencing some technical difficulties. Please try again in a few minutes, or call the clinic directly for assistance.", 
        sender: 'bot' 
      };
      setMessages(prev => [...prev, errorMessage]);
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="chat-window">
        <div className="chat-header">
          <h1>Clinic Assistant</h1>
        </div>
        
        <div className="chat-messages">
          {messages.map((msg) => (
            <div key={msg.id} className={`message-wrapper ${msg.sender}`}>
              <div className={`message ${msg.sender}`}>
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message-wrapper bot">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <form className="chat-input-area" onSubmit={handleSend}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
                e.preventDefault();
                handleSend(e);
              } else if (e.key === 'Enter' && (e.metaKey || e.ctrlKey || e.shiftKey)) {
                e.preventDefault();
                const start = e.target.selectionStart;
                const end = e.target.selectionEnd;
                setInput(input.substring(0, start) + '\n' + input.substring(end));
                setTimeout(() => {
                  e.target.selectionStart = e.target.selectionEnd = start + 1;
                }, 0);
              }
            }}
            placeholder="Type your message... (Cmd+Enter for new line)"
            disabled={isLoading}
            rows={2}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
