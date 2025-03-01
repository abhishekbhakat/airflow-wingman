document.addEventListener('DOMContentLoaded', function() {
    // Add title attributes for tooltips
    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(function(el) {
        el.title = el.getAttribute('title') || el.getAttribute('data-bs-original-title');
    });

    // Handle model selection and model name input
    const modelNameInput = document.getElementById('modelName');
    const modelRadios = document.querySelectorAll('input[name="model"]');

    modelRadios.forEach(function(radio) {
        radio.addEventListener('change', function() {
            const provider = this.value.split(':')[0];
            const modelName = this.getAttribute('data-model-name');
            console.log('Selected provider:', provider);
            console.log('Model name:', modelName);

            if (provider === 'openrouter') {
                console.log('Enabling model name input');
                modelNameInput.disabled = false;
                modelNameInput.value = '';
                modelNameInput.placeholder = 'Enter model name for OpenRouter';
            } else {
                console.log('Disabling model name input');
                modelNameInput.disabled = true;
                modelNameInput.value = modelName;
            }
        });
    });

    // Set initial state based on default selection
    const defaultSelected = document.querySelector('input[name="model"]:checked');
    if (defaultSelected) {
        const provider = defaultSelected.value.split(':')[0];
        const modelName = defaultSelected.getAttribute('data-model-name');
        console.log('Initial provider:', provider);
        console.log('Initial model name:', modelName);

        if (provider === 'openrouter') {
            console.log('Initially enabling model name input');
            modelNameInput.disabled = false;
            modelNameInput.value = '';
            modelNameInput.placeholder = 'Enter model name for OpenRouter';
        } else {
            console.log('Initially disabling model name input');
            modelNameInput.disabled = true;
            modelNameInput.value = modelName;
        }
    }

    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const refreshButton = document.getElementById('refresh-button');
    const chatMessages = document.getElementById('chat-messages');

    let currentMessageDiv = null;
    let messageHistory = [];

    // Create a processing indicator element
    const processingIndicator = document.createElement('div');
    processingIndicator.className = 'processing-indicator';
    processingIndicator.textContent = 'Processing tool calls...';
    chatMessages.appendChild(processingIndicator);

    function clearChat() {
        // Clear the chat messages
        chatMessages.innerHTML = '';
        // Add back the processing indicator
        chatMessages.appendChild(processingIndicator);
        // Reset message history
        messageHistory = [];
        // Clear the input field
        messageInput.value = '';
        // Enable input if it was disabled
        messageInput.disabled = false;
        sendButton.disabled = false;
    }

    function addMessage(content, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'message-user' : 'message-assistant'}`;

        // Apply pre-formatted class to preserve whitespace and newlines
        messageDiv.classList.add('pre-formatted');

        // Use innerText instead of textContent to preserve newlines
        messageDiv.innerText = content;

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return messageDiv;
    }

    function showProcessingIndicator() {
        processingIndicator.classList.add('visible');
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function hideProcessingIndicator() {
        processingIndicator.classList.remove('visible');
    }

    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        // Get selected model
        const selectedModel = document.querySelector('input[name="model"]:checked');
        if (!selectedModel) {
            alert('Please select a model');
            return;
        }

        const [provider, modelId] = selectedModel.value.split(':');
        const modelName = provider === 'openrouter' ? modelNameInput.value : modelId;

        // Clear input and add user message
        messageInput.value = '';
        addMessage(message, true);

        // Add user message to history
        messageHistory.push({
            role: 'user',
            content: message
        });

        // Use full message history for the request
        const messages = [...messageHistory];

        // Create assistant message div
        currentMessageDiv = addMessage('', false);

        // Get API key
        const apiKey = document.getElementById('api-key').value.trim();
        if (!apiKey) {
            alert('Please enter an API key');
            return;
        }

        // Disable input while processing
        messageInput.disabled = true;
        sendButton.disabled = true;

        // Get CSRF token
        const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
        if (!csrfToken) {
            alert('CSRF token not found. Please refresh the page.');
            return;
        }

        // Create request data
        const requestData = {
            provider: provider,
            model: modelName,
            messages: messages,
            api_key: apiKey,
            stream: true,
            temperature: 0.7
        };
        console.log('Sending request:', {...requestData, api_key: '***'});

        try {
            // Send request
            const response = await fetch('/wingman/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to get response');
            }

            // Process the streaming response
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullResponse = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.trim() === '') continue;

                    if (line.startsWith('data: ')) {
                        const content = line.slice(6); // Remove 'data: ' prefix

                        // Check for special events or end marker
                        if (content === '[DONE]') {
                            console.log('Stream complete');

                            // Add assistant's response to history
                            if (fullResponse) {
                                messageHistory.push({
                                    role: 'assistant',
                                    content: fullResponse
                                });
                            }
                            continue;
                        }

                        // Try to parse as JSON for special events
                        try {
                            const parsed = JSON.parse(content);

                            if (parsed.event === 'tool_processing_start') {
                                console.log('Tool processing started');
                                showProcessingIndicator();
                                continue;
                            }

                            if (parsed.event === 'tool_processing_complete') {
                                console.log('Tool processing completed');
                                hideProcessingIndicator();
                                continue;
                            }

                            // Handle the complete response event
                            if (parsed.event === 'complete_response') {
                                console.log('Received complete response from backend');
                                // Use the complete response from the backend
                                fullResponse = parsed.content;

                                // Update the display with the complete response
                                if (!currentMessageDiv.classList.contains('pre-formatted')) {
                                    currentMessageDiv.classList.add('pre-formatted');
                                }
                                currentMessageDiv.innerText = fullResponse;
                                continue;
                            }

                            // If we have JSON that's not a special event, it might be content
                            currentMessageDiv.textContent += JSON.stringify(parsed);
                            fullResponse += JSON.stringify(parsed);
                        } catch (e) {
                            // Not JSON, handle as normal content
                            // console.log('Received chunk:', JSON.stringify(content));

                            // Add to full response
                            fullResponse += content;

                            // Create a properly formatted display
                            if (!currentMessageDiv.classList.contains('pre-formatted')) {
                                currentMessageDiv.classList.add('pre-formatted');
                            }

                            // Always rebuild the entire content from the full response
                            currentMessageDiv.innerText = fullResponse;
                        }
                        // Scroll to bottom
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }
                }
            }
        } catch (error) {
            console.error('Error:', error);
            if (currentMessageDiv) {
                currentMessageDiv.textContent = `Error: ${error.message}`;
                currentMessageDiv.style.color = 'red';
            }
        } finally {
            // Always re-enable input and hide indicators
            messageInput.disabled = false;
            sendButton.disabled = false;
            hideProcessingIndicator();
        }
    }

    sendButton.addEventListener('click', sendMessage);
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    refreshButton.addEventListener('click', clearChat);
});
