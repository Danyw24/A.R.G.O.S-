const canvas = document.getElementById('waveCanvas');
const ctx = canvas.getContext('2d');
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const recordBtn = document.getElementById('recordBtn');
const fileInput = document.getElementById('fileInput');
const themeBtn = document.getElementById('theme-btn');
const waveContainer = document.querySelector('.wave-container');

let isRecording = false;
let mediaRecorder;
let audioChunks = [];
let audioContext, analyser;
let animationId;

// Configuración inicial
canvas.width = window.innerWidth;
canvas.height = 100;

// Event Listeners
sendBtn.addEventListener('click', handleInput);
userInput.addEventListener('keypress', e => e.key === 'Enter' && handleInput());
recordBtn.addEventListener('click', toggleRecording);
fileInput.addEventListener('change', handleFileUpload);
themeBtn.addEventListener('click', toggleTheme);

// Manejo de mensajes
async function handleInput() {
    const message = userInput.value.trim();
    if (!message) return;
    
    addMessage(message, 'user');
    userInput.value = '';
    
    try {
        const response = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer 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'
            },
            body: JSON.stringify({
                model: "gpt-3.5-turbo",
                messages: [{
                    role: "user",
                    content: message
                }],
                temperature: 0.7
            })
        });
        
        const data = await response.json();
        const aiResponse = data.choices[0].message.content;
        addMessage(aiResponse, 'assistant');
        speak(aiResponse);
    } catch (error) {
        console.error('Error:', error);
        addMessage('Error conectando con la IA', 'assistant');
        speak('Lo siento, hubo un error en mi procesamiento');
    }
}

// Tema claro/oscuro
function toggleTheme() {
    document.body.classList.toggle('light-mode');
    themeBtn.innerHTML = document.body.classList.contains('light-mode') 
        ? '<i class="fas fa-sun"></i>' 
        : '<i class="fas fa-moon"></i>';
}

function addMessage(content, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', `${sender}-message`);
    messageDiv.textContent = content;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function speak(text) {
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = synth.getVoices().find(voice => voice.name === 'Google UK English Male');
    utterance.onstart = () => waveContainer.classList.add('active');
    utterance.onend = () => waveContainer.classList.remove('active');
    synth.speak(utterance);
}

// Recorder (audio)
function toggleRecording() {
    if (isRecording) {
        mediaRecorder.stop();
    } else {
        startRecording();
    }
    isRecording = !isRecording;
    recordBtn.classList.toggle('recording', isRecording);
}

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);
                audio.play();
                audioChunks = [];
            };
            mediaRecorder.start();
        })
        .catch(error => console.error('Error al grabar audio:', error));
}

// File upload
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = () => {
            const fileContent = reader.result;
            addMessage(`Archivo recibido: ${file.name}`, 'assistant');
        };
        reader.readAsText(file);
    }
}
