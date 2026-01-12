// Utility function to split long messages into multiple parts based on word count
export const splitMessage = (text, maxWords = 24) => {
  const words = text.trim().split(/\s+/);
  
  if (words.length <= maxWords) {
    return [text];
  }

  // Split by complete sentences only (Persian and English punctuation)
  const sentenceEndings = /[.!?؟]\s+/g;
  const sentences = text.split(sentenceEndings);
  const sentenceEndingsFound = text.match(sentenceEndings) || [];
  
  // Reconstruct sentences with their endings
  const fullSentences = [];
  for (let i = 0; i < sentences.length; i++) {
    if (sentences[i].trim()) {
      const ending = sentenceEndingsFound[i] || '';
      fullSentences.push(sentences[i].trim() + ending);
    }
  }

  const parts = [];
  let currentPart = '';

  // Group sentences together while respecting word limit
  for (const sentence of fullSentences) {
    const sentenceWords = sentence.trim().split(/\s+/);
    const currentWords = currentPart.trim().split(/\s+/);
    
    if (currentWords.length + sentenceWords.length <= maxWords) {
      currentPart += (currentPart ? ' ' : '') + sentence;
    } else {
      if (currentPart.trim()) {
        parts.push(currentPart.trim());
      }
      currentPart = sentence;
    }
  }

  if (currentPart.trim()) {
    parts.push(currentPart.trim());
  }

  // If we still have parts that are too long, ONLY split at sentence boundaries
  // Never split in the middle of a sentence
  const finalParts = [];
  for (const part of parts) {
    const partWords = part.trim().split(/\s+/);
    
    if (partWords.length <= maxWords) {
      finalParts.push(part);
    } else {
      // If a single sentence is too long, we have to keep it as one message
      // This is better than breaking it in the middle
      finalParts.push(part);
    }
  }

  return finalParts;
};

// Function to add messages with delay
export const addMessagesWithDelay = async (setMessages, messages, delayMs = 1000) => {
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    
    if (i === 0) {
      // Add first message immediately
      setMessages(prev => [...prev, message]);
    } else {
      // Add subsequent messages with delay
      await new Promise(resolve => setTimeout(resolve, delayMs));
      setMessages(prev => [...prev, message]);
    }
  }
};
