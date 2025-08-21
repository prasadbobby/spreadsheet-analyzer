"use client";

import React, {
  useState,
  useRef,
  useEffect,
  useMemo,
  useCallback,
} from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Send,
  Download,
  Trash2,
  Brain,
  User,
  Database,
  BarChart3,
  Search,
  Lightbulb,
  Table,
  Loader2,
  Sparkles,
  TrendingUp,
  Eye,
  Zap,
  Copy,
  MessageSquare,
} from "lucide-react";
import { useAISheetChat } from "@/contexts/AISheetChatContext";
import { formatMessageText, getCurrentTime } from "@/lib/utils";
import WelcomeScreen from "@/components/WelcomeScreen";
import DataTable from "@/components/DataTable";
import ChartVisualization from "@/components/ChartVisualization";
import ClearSessionDialog from "@/components/ClearSessionDialog";

// Memoized components to prevent unnecessary re-renders
const MemoizedDataTable = React.memo(DataTable);
const MemoizedChartVisualization = React.memo(ChartVisualization);

export default function MainContent() {
  const {
    currentChatId,
    datasetLoaded,
    isProcessing,
    messages,
    sendMessage,
    clearChat,
  } = useAISheetChat();

  const [inputMessage, setInputMessage] = useState("");
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const [clearDialogOpen, setClearDialogOpen] = useState(false);

  // Debounced input to reduce re-renders
  const [debouncedInputMessage, setDebouncedInputMessage] = useState("");
  const debounceTimeoutRef = useRef(null);

  // Optimized scroll function with throttling
  const scrollToBottom = useCallback(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({
        behavior: "smooth",
        block: "end",
      });
    }
  }, []);

  // Throttled scroll effect
  useEffect(() => {
    const timeoutId = setTimeout(scrollToBottom, 100);
    return () => clearTimeout(timeoutId);
  }, [messages.length, scrollToBottom]);

  // Debounced input handling
  useEffect(() => {
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }

    debounceTimeoutRef.current = setTimeout(() => {
      setDebouncedInputMessage(inputMessage);
    }, 150); // 150ms debounce

    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, [inputMessage]);

  const handleSend = useCallback(async () => {
    if (
      inputMessage.trim() &&
      !isProcessing &&
      datasetLoaded &&
      currentChatId
    ) {
      const message = inputMessage.trim();
      setInputMessage("");

      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }

      await sendMessage(message);
    }
  }, [inputMessage, isProcessing, datasetLoaded, currentChatId, sendMessage]);

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  // Optimized auto-resize with throttling
  const autoResize = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
    }
  }, []);

  // Throttled input change handler
  const handleInputChange = useCallback(
    (e) => {
      setInputMessage(e.target.value);
      // Throttle auto-resize
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
      debounceTimeoutRef.current = setTimeout(autoResize, 50);
    },
    [autoResize]
  );

  const setQuickMessage = useCallback(
    (message) => {
      if (datasetLoaded && !isProcessing && currentChatId) {
        setInputMessage(message);
        textareaRef.current?.focus();
        setTimeout(autoResize, 0);
      }
    },
    [datasetLoaded, isProcessing, currentChatId, autoResize]
  );

  const handleClearSession = useCallback(() => {
    clearChat();
  }, [clearChat]);

  // Memoized quick actions to prevent re-creation
  const quickActions = useMemo(
    () => [
      {
        icon: Eye,
        label: "View All Data",
        message: "Show me all the data",
        color: "bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100",
      },
      {
        icon: TrendingUp,
        label: "Analyze Trends",
        message: "Analyze this dataset for patterns and trends",
        color: "bg-green-50 text-green-700 border-green-200 hover:bg-green-100",
      },
      {
        icon: BarChart3,
        label: "Create Chart",
        message: "Visualize this data with a chart",
        color:
          "bg-purple-50 text-purple-700 border-purple-200 hover:bg-purple-100",
      },
      {
        icon: Search,
        label: "Find Insights",
        message: "What are the most interesting insights in this data?",
        color:
          "bg-orange-50 text-orange-700 border-orange-200 hover:bg-orange-100",
      },
      {
        icon: Lightbulb,
        label: "Get Summary",
        message: "Give me a comprehensive summary of this dataset",
        color:
          "bg-yellow-50 text-yellow-700 border-yellow-200 hover:bg-yellow-100",
      },
    ],
    []
  );

  // Memoized helper functions
  const getAnalysisIcon = useCallback((analysisType) => {
    const icons = {
      mcp_analysis: Database,
      ai_analysis: Brain,
      data_retrieval: Table,
      summary: BarChart3,
      visualization: TrendingUp,
      direct_analysis: Lightbulb,
      overview: Eye,
      patterns: Search,
    };
    return icons[analysisType] || Sparkles;
  }, []);

  const getAnalysisLabel = useCallback((analysisType) => {
    const labels = {
      mcp_analysis: "Data Analysis",
      ai_analysis: "AI Insights",
      data_retrieval: "Data Query",
      summary: "Summary",
      visualization: "Visualization",
      direct_analysis: "Analysis",
      overview: "Overview",
      patterns: "Pattern Recognition",
    };
    return labels[analysisType] || "AI Response";
  }, []);

  const getAnalysisColor = useCallback((analysisType) => {
    const colors = {
      mcp_analysis: "bg-blue-100 text-blue-800",
      ai_analysis: "bg-purple-100 text-purple-800",
      data_retrieval: "bg-green-100 text-green-800",
      summary: "bg-orange-100 text-orange-800",
      visualization: "bg-pink-100 text-pink-800",
      direct_analysis: "bg-yellow-100 text-yellow-800",
      overview: "bg-indigo-100 text-indigo-800",
      patterns: "bg-teal-100 text-teal-800",
    };
    return colors[analysisType] || "bg-gray-100 text-gray-800";
  }, []);

  const copyToClipboard = useCallback((text) => {
    navigator.clipboard.writeText(text);
  }, []);

  // Memoized messages to prevent unnecessary re-renders
  const memoizedMessages = useMemo(() => {
    return messages.map((message, index) => (
      <MessageBubble
        key={message.id}
        message={message}
        getAnalysisIcon={getAnalysisIcon}
        getAnalysisLabel={getAnalysisLabel}
        getAnalysisColor={getAnalysisColor}
        copyToClipboard={copyToClipboard}
      />
    ));
  }, [
    messages,
    getAnalysisIcon,
    getAnalysisLabel,
    getAnalysisColor,
    copyToClipboard,
  ]);

  // Show welcome screen if no current chat
  if (!currentChatId) {
    return (
      <div className="flex-1 flex flex-col bg-gray-50 min-h-0">
        <div className="border-b border-gray-200 bg-white shadow-sm flex-shrink-0 h-[120px] flex items-center">
          <div className="p-6 w-full">
            <div className="flex justify-between items-center">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  AI Data Analysis
                </h1>
                <p className="text-sm text-gray-600 mt-1">
                  Create a new session to start analyzing your data
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-6 max-w-md">
            <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto">
              <MessageSquare className="h-10 w-10 text-purple-600" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900 mb-2">
                Welcome to AI Sheet Chat
              </h2>
              <p className="text-gray-600 mb-4">
                Create a new analysis session to upload your data and start
                asking questions.
              </p>
              <div className="text-sm text-gray-500">
                💡 Each session maintains its own dataset and conversation
                history
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="flex-1 flex flex-col bg-gray-50 min-h-0">
        {/* Header - Match Sidebar Height */}
        <div className="border-b border-gray-200 bg-white shadow-sm flex-shrink-0 h-[120px] flex items-center">
          <div className="p-6 w-full">
            <div className="flex justify-between items-center">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  AI Data Analysis
                </h1>
                <p className="text-sm text-gray-600 mt-1">
                  {datasetLoaded
                    ? "Ask questions about your data in natural language"
                    : "Upload data to this session to start analyzing"}
                </p>
              </div>
              <div className="flex gap-3">
                <Button
                  variant="outline"
                  size="sm"
                  className="shadow-sm hover:shadow-md"
                >
                  <Download className="h-4 w-4 mr-2" />
                  Export Session
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setClearDialogOpen(true)}
                  disabled={isProcessing || messages.length === 0}
                  className="shadow-sm hover:bg-red-50 hover:border-red-200 hover:text-red-700"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Clear Session
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Messages - Optimized rendering */}
        <div className="flex-1 min-h-0">
          <ScrollArea className="h-full custom-scrollbar">
            {messages.length === 0 ? (
              <WelcomeScreen />
            ) : (
              <div className="p-6 space-y-8 max-w-6xl mx-auto">
                {memoizedMessages}

                {/* AI Thinking Indicator */}
                {isProcessing && (
                  <div className="flex gap-6 justify-start fade-in">
                    <div className="flex-shrink-0">
                      <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
                        <Brain className="h-5 w-5 text-white" />
                      </div>
                    </div>
                    <div className="bg-white border border-gray-200 rounded-2xl shadow-lg p-6 max-w-[75%]">
                      <div className="flex items-center gap-3">
                        <Loader2 className="h-5 w-5 animate-spin text-purple-600" />
                        <div>
                          <div className="text-sm font-medium text-gray-900 mb-1">
                            AI is analyzing your data...
                          </div>
                          <div className="flex items-center gap-2 text-xs text-gray-500">
                            <Database className="h-3 w-3" />
                            Processing with session-specific dataset
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </ScrollArea>
        </div>

        {/* Input Area - Optimized for performance */}
        <div className="border-t border-gray-200 bg-white shadow-lg flex-shrink-0">
          <div className="p-6">
            <div className="max-w-4xl mx-auto space-y-4">
              {/* Quick Actions - Only show when needed */}
              {datasetLoaded && messages.length === 1 && (
                <div className="mb-4">
                  <div className="flex items-center gap-2 mb-3">
                    <Zap className="h-4 w-4 text-purple-600" />
                    <span className="text-sm font-medium text-gray-700">
                      Quick Actions
                    </span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
                    {quickActions.map((action, index) => (
                      <Button
                        key={index}
                        variant="outline"
                        size="sm"
                        onClick={() => setQuickMessage(action.message)}
                        disabled={
                          !datasetLoaded || isProcessing || !currentChatId
                        }
                        className={`text-xs h-auto p-4 flex flex-col items-center gap-2 hover:shadow-md transition-all ${action.color}`}
                      >
                        <action.icon className="h-5 w-5" />
                        <span className="font-medium">{action.label}</span>
                      </Button>
                    ))}
                  </div>
                </div>
              )}

              {/* Optimized Input Field */}
              <div className="flex gap-4 items-end">
                <div className="flex-1">
                  <Textarea
                    ref={textareaRef}
                    value={inputMessage}
                    onChange={handleInputChange}
                    onKeyDown={handleKeyDown}
                    placeholder={
                      !currentChatId
                        ? "Create or select a chat session first..."
                        : !datasetLoaded
                        ? "Upload your dataset to this session first to start analyzing..."
                        : "Ask me anything about your data - I can analyze patterns, create insights, and answer complex questions..."
                    }
                    disabled={!datasetLoaded || isProcessing || !currentChatId}
                    className="min-h-[60px] max-h-[120px] resize-none border-gray-200 focus:border-purple-500 focus:ring-purple-500 rounded-xl shadow-sm text-sm"
                    rows={1}
                  />
                </div>
                <Button
                  onClick={handleSend}
                  disabled={
                    !datasetLoaded ||
                    isProcessing ||
                    !inputMessage.trim() ||
                    !currentChatId
                  }
                  size="lg"
                  className="h-[60px] px-6 shadow-lg hover:shadow-xl transition-all rounded-xl"
                  style={{ backgroundColor: "#8500df" }}
                >
                  {isProcessing ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Send className="h-5 w-5" />
                  )}
                </Button>
              </div>

              {/* Helper Text */}
              {datasetLoaded && currentChatId && (
                <div className="text-center">
                  <p className="text-xs text-gray-500">
                    💡 Try: &quot;Create a bar chart&quot;, &quot;Show sales by
                    department&quot;, &quot;Find unusual patterns&quot;, or
                    &quot;Summarize key insights&quot;
                  </p>
                </div>
              )}

              {!currentChatId && (
                <div className="text-center">
                  <p className="text-xs text-gray-500">
                    📝 Create a new analysis session from the sidebar to get
                    started
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Clear Session Dialog */}
      <ClearSessionDialog
        open={clearDialogOpen}
        onOpenChange={setClearDialogOpen}
        onConfirm={handleClearSession}
      />
    </>
  );
}

// Memoized Message Component
const MessageBubble = React.memo(
  ({
    message,
    getAnalysisIcon,
    getAnalysisLabel,
    getAnalysisColor,
    copyToClipboard,
  }) => {
    return (
      <div
        className={`flex gap-6 ${
          message.type === "user" ? "justify-end" : "justify-start"
        } fade-in`}
      >
        {/* Assistant Avatar - Left Side */}
        {message.type === "assistant" && (
          <div className="flex-shrink-0">
            <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center shadow-lg">
              <Brain className="h-5 w-5 text-white" />
            </div>
          </div>
        )}

        {/* Message Content */}
        <div
          className={`max-w-[75%] ${message.type === "user" ? "order-1" : ""}`}
        >
          <div
            className={`rounded-2xl shadow-lg ${
              message.type === "user"
                ? "text-white ml-auto"
                : "bg-white border border-gray-200"
            }`}
            style={
              message.type === "user" ? { backgroundColor: "#8500df" } : {}
            }
          >
            {/* Message Header */}
            <div className="px-6 py-4 border-b border-gray-100">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <span
                    className={`text-sm font-semibold ${
                      message.type === "user" ? "text-white" : "text-gray-900"
                    }`}
                  >
                    {message.type === "user" ? "You" : "AI Assistant"}
                  </span>
                  {message.type === "assistant" && message.analysisType && (
                    <Badge
                      className={`text-xs ${getAnalysisColor(
                        message.analysisType
                      )}`}
                    >
                      {(() => {
                        const Icon = getAnalysisIcon(message.analysisType);
                        return (
                          <div className="flex items-center gap-1">
                            <Icon className="h-3 w-3" />
                            {getAnalysisLabel(message.analysisType)}
                          </div>
                        );
                      })()}
                    </Badge>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  <span
                    className={`text-xs ${
                      message.type === "user"
                        ? "text-white/80"
                        : "text-gray-500"
                    }`}
                  >
                    {getCurrentTime()}
                  </span>
                  {message.type === "assistant" && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 hover:bg-gray-100"
                      onClick={() => copyToClipboard(message.content)}
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                  )}
                </div>
              </div>
            </div>

            {/* Message Content */}
            <div className="px-6 py-4">
              <div
                className={`prose prose-sm max-w-none ${
                  message.type === "user" ? "prose-invert" : ""
                }`}
                dangerouslySetInnerHTML={{
                  __html: formatMessageText(message.content),
                }}
              />

              {/* SQL Query Display */}
              {message.query && message.type === "assistant" && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Database className="h-4 w-4 text-gray-600" />
                      <span className="text-sm font-medium text-gray-700">
                        Generated SQL Query
                      </span>
                    </div>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 w-6 p-0"
                      onClick={() => copyToClipboard(message.query)}
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                  </div>
                  <code className="text-xs font-mono text-gray-800 bg-white p-3 rounded border block overflow-x-auto whitespace-pre-wrap">
                    {message.query}
                  </code>
                </div>
              )}

              {/* Chart Visualization */}
              {message.chartData && (
                <div className="mt-4">
                  <MemoizedChartVisualization
                    chartData={message.chartData}
                    title="Data Visualization"
                  />
                </div>
              )}

              {/* Data Table */}
              {message.result &&
                Array.isArray(message.result) &&
                message.result.length > 0 && (
                  <div className="mt-4">
                    <MemoizedDataTable data={message.result} />
                  </div>
                )}
            </div>
          </div>
        </div>

        {/* User Avatar - Right Side */}
        {message.type === "user" && (
          <div className="flex-shrink-0 order-2">
            <div
              className="w-10 h-10 rounded-full flex items-center justify-center shadow-lg"
              style={{ backgroundColor: "#8500df" }}
            >
              <User className="h-5 w-5 text-white" />
            </div>
          </div>
        )}
      </div>
    );
  }
);

MessageBubble.displayName = "MessageBubble";
