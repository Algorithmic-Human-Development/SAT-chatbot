import React, { useState, useRef, useEffect } from "react";
import { FiSend, FiMic, FiStopCircle, FiRefreshCw } from "react-icons/fi";
import { CiLogout } from "react-icons/ci";
import { TbMessageChatbot, TbMessageChatbotFilled } from "react-icons/tb";
import axios from "axios";
import { message } from "antd";
import clsx from "clsx";
import "./App.css";
import { splitMessage, addMessagesWithDelay } from "./utils/messageSplitter";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const ChatbotPlacebo = () => {
    const [messages, setMessages] = useState([
        { text: "سلام! خوشحالم میبینمت.", sender: "bot" },
    ]);
    const [input, setInput] = useState("");
    const [isTyping, setIsTyping] = useState(false);
    const [recording, setRecording] = useState(false);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef(null);
    const [messageApi, contextHolder] = message.useMessage();
    const messagesEndRef = useRef(null);
    const chatAreaRef = useRef(null);

    const baseURL = process.env.REACT_APP_BASE_URL;

    useEffect(() => {
        if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages]);

    const sendMessage = async () => {
        const text = input.trim();
        if (!text) return;
        const userMessage = { text, sender: "user" };
        setMessages((prevMessages) => [...prevMessages, userMessage]);
        setInput("");
        setIsTyping(true);

        try {
            const response = await axios.post(
                `${baseURL}/api/placebo-chat/`,
                { text },
                {
                    headers: {
                        Authorization: `Bearer ${localStorage.getItem("access")}`,
                    },
                }
            );

            let botResponse = response.data.response;

            // For placebo version, we don't have exercises or complex states
            // Just split the message if it's long
            const messageParts = splitMessage(botResponse);
            const botMessages = messageParts.map((part) => ({
                text: part,
                sender: "bot",
            }));

            await addMessagesWithDelay(setMessages, botMessages, 1000);
        } catch (error) {
            if (
                error.response &&
                (error.response.status === 401 || error.response.data?.detail === "Invalid token")
            ) {
                localStorage.clear();
                sessionStorage.clear();
                setMessages((prev) => [
                    ...prev,
                    { text: "Session expired. Please log in again.", sender: "bot" },
                ]);
            } else {
                setMessages((prev) => [
                    ...prev,
                    { text: "Error: Unable to fetch response.", sender: "bot" },
                ]);
            }
        } finally {
            setIsTyping(false);
        }
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            setRecording(true);
            audioChunksRef.current = [];
            const mediaRecorder = new window.MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };
            mediaRecorder.onstop = async () => {
                setRecording(false);
                const audioBlob = new Blob(audioChunksRef.current, {
                    type: "audio/wav",
                });
                sendAudioMessage(audioBlob);
            };
            mediaRecorder.start();
        } catch (error) {
            messageApi.open({
                type: "error",
                content:
                    "نمیتوان به میکروفن دسترسی پیدا کرد. لطفاً تنظیمات مرورگر را بررسی کنید.",
            });
            setRecording(false);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();
        }
    };

    const sendAudioMessage = async (audioBlob) => {
        const formData = new FormData();
        formData.append("audio", audioBlob, "voice_message.wav");

        try {
            const response = await axios.post(
                `${baseURL}/api/placebo-chat/`,
                formData,
                {
                    headers: {
                        Authorization: `Bearer ${localStorage.getItem("access")}`,
                        "Content-Type": "multipart/form-data",
                    },
                }
            );

            let botResponse = response.data.response;

            // For placebo version, just split the message if it's long
            const messageParts = splitMessage(botResponse);
            const botMessages = messageParts.map((part) => ({
                text: part,
                sender: "bot",
            }));

            await addMessagesWithDelay(setMessages, botMessages, 1000);
        } catch (error) {
            setMessages((prev) => [
                ...prev,
                { text: "خطا در ارسال پیام صوتی.", sender: "user" },
            ]);
        }
    };

    const handleLogout = () => {
        localStorage.clear();
        window.location.reload();
    };

    const handleRestart = async () => {
        try {
            // Call the reset API endpoint - placebo version might not have this endpoint
            // but we'll try it and handle gracefully if it fails
            await axios.post(
                `${baseURL}/api/reset-state/`,
                {},
                {
                    headers: {
                        Authorization: `Bearer ${localStorage.getItem("access")}`,
                    },
                }
            );

            // Reset local state
            setMessages([{ text: "سلام! خوشحالم میبینمت.", sender: "bot" }]);
            setInput("");

            // Show success message
            messageApi.success("دوباره شروع کنیم");
        } catch (error) {
            console.error("Error resetting state:", error);
            messageApi.info("چت ریست شد");

            // Reset local state even if API call fails
            setMessages([{ text: "سلام! خوشحالم میبینمت.", sender: "bot" }]);
            setInput("");
        }
    };

    return (
        <>
            {contextHolder}
            <div className="flex items-center justify-center h-screen bg-gradient-to-br from-purple-100 to-blue-50">
                <div className="flex flex-col w-full max-w-3xl h-full bg-grey-50 shadow-2xl rounded-lg overflow-hidden">
                    <HeaderComponent
                        handleLogout={handleLogout}
                        handleRestart={handleRestart}
                    />
                    <div
                        ref={chatAreaRef}
                        className="flex-grow overflow-y-auto p-4 space-y-1 app-background"
                    >
                        {messages.map((msg, index) => (
                            <MessageComponent
                                key={index}
                                text={msg.text}
                                sender={msg.sender === "user" ? "me" : "bot"}
                                image={msg.image}
                            />
                        ))}
                        <div ref={messagesEndRef} />
                        {isTyping && (
                            <div className="flex bg-grey-600 justify-start">
                                <div className="px-4 py-2 bg-gray-200 rounded-r-2xl rounded-tl-2xl">
                                    <div className="flex space-x-1">
                                        <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                                        <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                                        <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Simple restart button - always available for placebo version */}
                    <div className="flex justify-center py-2">
                        <button
                            onClick={handleRestart}
                            className="bg-orange-500 hover:bg-orange-600 text-white px-6 py-2 rounded-full shadow-md transition duration-200 flex items-center space-x-2"
                        >
                            <FiRefreshCw size={16} />
                            <span>شروع دوباره</span>
                        </button>
                    </div>

                    <div className="flex items-center px-4 py-3 bg- shadow-lg">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            className="flex-grow px-4 py-2 border border-gray-300 rounded-full"
                            placeholder="پیام خود را وارد کنید..."
                            onKeyDown={(e) => {
                                if (e.key === "Enter") {
                                    sendMessage();
                                }
                            }}
                            dir="rtl"
                        />
                        {recording && (
                            <div dir="rtl" className="flex justify-start">
                                <div className="px-4 py-2 bg-gray-200 rounded-r-2xl rounded-tl-2xl flex items-center space-x-2">
                                    <span className="w-3 h-3 bg-red-500 rounded-full animate-ping"></span>
                                    <span className="text-sm text-red-600">
                                        در حال ضبط صدا...
                                    </span>
                                </div>
                            </div>
                        )}
                        {/* Voice mic button hidden per request
                        <button
                            onClick={recording ? stopRecording : startRecording}
                            className="ml-2 p-3 rounded-full bg-red-500 text-white"
                        >
                            {recording ? <FiStopCircle size={20} /> : <FiMic size={20} />}
                        </button>
                        */}
                        <button
                            onClick={sendMessage}
                            className="ml-2 p-3 rounded-full bg-blue-500 text-white"
                        >
                            <FiSend size={20} />
                        </button>
                    </div>
                </div>
            </div>
        </>
    );
};

const HeaderComponent = ({ handleLogout, handleRestart }) => {
    return (
        <div className="flex items-center justify-between p-4 bg-gradient-to-r from-blue-500 to-indigo-500 text-white shadow-md">
            <div className="flex items-center space-x-2">
                <button
                    className="hover:text-red-400 transition duration-200"
                    onClick={handleLogout}
                >
                    <CiLogout size={24} />
                </button>
                <button
                    className="hover:text-yellow-400 transition duration-200"
                    onClick={handleRestart}
                    title="شروع مجدد"
                >
                    <FiRefreshCw size={24} />
                </button>
            </div>
            <h2 className="text-xl font-bold">بات دلبستگی به خود</h2>
            <div className="rounded-full bg-teal-800 p-2 shadow-md">
                <TbMessageChatbotFilled className="text-2xl" />
            </div>
        </div>
    );
};

const MessageComponent = ({ text, sender, name, image }) => {
    const isMe = sender === "me";

    return (
        <div
            className={clsx(
                "flex w-full items-end space-x-2 my-2",
                isMe ? "justify-end" : "justify-start"
            )}
        >
            {!isMe && <TbMessageChatbot className="w-8 h-8 mb-8" />}

            <div
                className={clsx(
                    "max-w-xs sm:max-w-md",
                    isMe ? "text-right" : "text-left"
                )}
                dir="rtl"
            >
                {name && !isMe && (
                    <div className="text-xs text-gray-500 font-medium mb-1 ml-1">
                        {name}
                    </div>
                )}

                {image && (
                    <img
                        src={image.src}
                        alt={image.alt}
                        className="mt-2 rounded-lg max-w-full -48 object-contain"
                    />
                )}

                <div
                    className={clsx(
                        "rounded-2xl px-4 py-2 shadow-md text-lg break-words paragraph",
                        isMe
                            ? "bg-blue-500 text-white rounded-br-none"
                            : "bg-slate-200 text-slate-800 rounded-r-2xl rounded-tl-2xl max-w-m px-4 py-2"
                    )}
                >
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                            p: ({ node, ...props }) => (
                                <p
                                    style={{ marginBottom: "0.5rem", whiteSpace: "pre-wrap", fontSize: "1.125rem" }}
                                    {...props}
                                />
                            ),
                            li: ({ node, ...props }) => (
                                <li style={{ marginBottom: "0.3rem", fontSize: "1.125rem" }} {...props} />
                            ),
                        }}
                    >
                        {text}
                    </ReactMarkdown>
                </div>
            </div>
        </div>
    );
};

export default ChatbotPlacebo;
