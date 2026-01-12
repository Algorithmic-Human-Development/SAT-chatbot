import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { DownOutlined, UpOutlined } from '@ant-design/icons';

export default function ExerciseMessage({ msg, img }) {
    const [isExplanationExpanded, setIsExplanationExpanded] = useState(false);
    const [isSolutionExpanded, setIsSolutionExpanded] = useState(false);

    // Split the message into parts (question, explanation, solution)
    const parts = msg.split(/(### Explanation|### Solution)/);
    
    let question = parts[0];
    let explanation = '';
    let solution = '';

    if (parts.length > 1) {
        if (parts[1] === '### Explanation' && parts[2]) {
            explanation = parts[2];
            if (parts[3] === '### Solution' && parts[4]) {
                solution = parts[4];
            }
        } else if (parts[1] === '### Solution' && parts[2]) {
            solution = parts[2];
        }
    }

    return (
        <div className="flex justify-start">
            <div
                dir="rtl"
                className="max-w-xs bg-gray-200 text-blue-700 rounded-r-2xl rounded-tl-2xl p-4"
            >
                <img className="w-full rounded-lg mb-3 shadow-sm" alt="exercise" src={img} />

                {/* Question Section */}
                <div className="mb-4">
                    <ReactMarkdown 
                        remarkPlugins={[remarkGfm]}
                        className="text-lg leading-relaxed prose max-w-none"
                        components={{
                            // Custom styling for markdown elements
                            p: ({ children }) => <p className="mb-2 text-lg">{children}</p>,
                            strong: ({ children }) => <strong className="font-bold text-blue-800">{children}</strong>,
                            h1: ({ children }) => <h1 className="text-xl font-bold mb-2">{children}</h1>,
                            h2: ({ children }) => <h2 className="text-lg font-bold mb-2">{children}</h2>,
                            h3: ({ children }) => <h3 className="text-lg font-bold mb-1">{children}</h3>,
                            ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
                            ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
                            li: ({ children }) => <li className="text-lg">{children}</li>,
                            code: ({ children }) => <code className="bg-gray-300 px-1 py-0.5 rounded text-base">{children}</code>,
                            pre: ({ children }) => <pre className="bg-gray-300 p-2 rounded text-base overflow-x-auto">{children}</pre>,
                        }}
                    >
                        {question}
                    </ReactMarkdown>
                </div>

                {/* Explanation Section (Collapsible) */}
                {explanation && (
                    <div className="mb-3">
                        <button
                            onClick={() => setIsExplanationExpanded(!isExplanationExpanded)}
                            className="flex items-center justify-between w-full text-left text-sm font-medium text-blue-800 hover:text-blue-900 transition-colors"
                        >
                            <span>توضیحات</span>
                            {isExplanationExpanded ? <UpOutlined /> : <DownOutlined />}
                        </button>
                        
                        {isExplanationExpanded && (
                            <div className="mt-2 p-3 bg-blue-50 rounded-lg border border-blue-200">
                                <ReactMarkdown 
                                    remarkPlugins={[remarkGfm]}
                                    className="text-lg leading-relaxed prose max-w-none"
                                    components={{
                                        p: ({ children }) => <p className="mb-2 text-lg">{children}</p>,
                                        strong: ({ children }) => <strong className="font-bold text-blue-800">{children}</strong>,
                                        ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
                                        ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
                                        li: ({ children }) => <li className="text-lg">{children}</li>,
                                        code: ({ children }) => <code className="bg-blue-200 px-1 py-0.5 rounded text-base">{children}</code>,
                                    }}
                                >
                                    {explanation}
                                </ReactMarkdown>
                            </div>
                        )}
                    </div>
                )}

                {/* Solution Section (Collapsible) */}
                {solution && (
                    <div className="mb-3">
                        <button
                            onClick={() => setIsSolutionExpanded(!isSolutionExpanded)}
                            className="flex items-center justify-between w-full text-left text-sm font-medium text-green-800 hover:text-green-900 transition-colors"
                        >
                            <span>راه حل</span>
                            {isSolutionExpanded ? <UpOutlined /> : <DownOutlined />}
                        </button>
                        
                        {isSolutionExpanded && (
                            <div className="mt-2 p-3 bg-green-50 rounded-lg border border-green-200">
                                <ReactMarkdown 
                                    remarkPlugins={[remarkGfm]}
                                    className="text-lg leading-relaxed prose max-w-none"
                                    components={{
                                        p: ({ children }) => <p className="mb-2 text-lg">{children}</p>,
                                        strong: ({ children }) => <strong className="font-bold text-green-800">{children}</strong>,
                                        ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1">{children}</ul>,
                                        ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1">{children}</ol>,
                                        li: ({ children }) => <li className="text-lg">{children}</li>,
                                        code: ({ children }) => <code className="bg-green-200 px-1 py-0.5 rounded text-base">{children}</code>,
                                    }}
                                >
                                    {solution}
                                </ReactMarkdown>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
