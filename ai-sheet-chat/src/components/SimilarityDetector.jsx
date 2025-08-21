'use client'

import { useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Input } from '@/components/ui/input'
import { 
  Brain, 
  Search, 
  GitMerge, 
  Target, 
  TrendingUp,
  CheckCircle,
  Clock,
  Hash,
  Zap,
  RefreshCw
} from 'lucide-react'
import { useAISheetChat } from '@/contexts/AISheetChatContext'
import VirtualizedDataTable from './VirtualizedDataTable'

export default function SimilarityDetector() {
  const { currentChatId, datasetLoaded, sendMessage } = useAISheetChat()
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [similarityResults, setSimilarityResults] = useState(null)
  const [threshold, setThreshold] = useState(0.7)
  const [searchQuery, setSearchQuery] = useState('')
  const [analysisType, setAnalysisType] = useState('comprehensive')

  const analyzeSimilarity = useCallback(async () => {
    if (!datasetLoaded || !currentChatId) return

    setIsAnalyzing(true)
    setProgress(0)
    setSimilarityResults(null)

    try {
      // Progress simulation
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90))
      }, 500)

      const analysisQuery = generateAnalysisQuery(analysisType, threshold, searchQuery)
      
      // Use the chat system for analysis
      await sendMessage(analysisQuery)
      
      clearInterval(progressInterval)
      setProgress(100)

      // Simulate results for demo (in real app, results would come from sendMessage response)
      setTimeout(() => {
        setSimilarityResults({
          data: [], // This would be populated by actual analysis
          summary: `Analysis completed using ${analysisType} method with ${threshold * 100}% threshold.`,
          total_matches: 0
        })
        setProgress(0)
      }, 2000)

    } catch (error) {
      console.error('Similarity analysis error:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }, [datasetLoaded, currentChatId, analysisType, threshold, searchQuery, sendMessage])

  const generateAnalysisQuery = (type, threshold, query) => {
    const baseQuery = query || "analyze this dataset"
    
    switch (type) {
      case 'duplicates':
        return `Find duplicate and near-duplicate records in this dataset with ${threshold * 100}% similarity threshold. Look for exact matches and semantic duplicates across all text fields. Show: 1) Exact duplicates, 2) Near-duplicates with similarity scores, 3) Potential data quality issues, 4) Recommended cleanup actions. Group similar records together and highlight key differences.`
      
      case 'semantic':
        return `Perform semantic similarity analysis on "${baseQuery}". Use text embedding and cosine similarity to: 1) Find semantically similar records even with different wording, 2) Identify content clusters and themes, 3) Detect related items based on meaning rather than exact text matches, 4) Show similarity scores and reasoning. Focus on understanding context and meaning.`
      
      case 'categorical':
        return `Analyze and categorize records in this dataset related to "${baseQuery}". Perform: 1) Automatic categorization of similar items, 2) Pattern recognition across categories, 3) Anomaly detection within categories, 4) Priority scoring based on similarity patterns, 5) Trend analysis across different groups. Create smart categories and show distribution.`
      
      case 'comprehensive':
      default:
        return `Perform comprehensive similarity analysis on this dataset related to "${baseQuery}". Include: 1) **Duplicate Detection**: Find exact and near-duplicate records, 2) **Semantic Similarity**: Identify related content based on meaning with ${threshold * 100}% similarity threshold, 3) **Smart Categorization**: Group similar items automatically, 4) **Pattern Analysis**: Detect trends and anomalies, 5) **Priority Scoring**: Rank by similarity and importance, 6) **Recommendations**: Suggest data cleanup and organization improvements. Provide detailed similarity scores and explanations.`
    }
  }

  const analysisTypes = [
    { 
      id: 'comprehensive', 
      label: 'Comprehensive Analysis', 
      icon: Brain,
      description: 'Full similarity analysis with duplicates, semantic matching, and categorization'
    },
    { 
      id: 'duplicates', 
      label: 'Duplicate Detection', 
      icon: GitMerge,
      description: 'Find exact and near-duplicate records'
    },
    { 
      id: 'semantic', 
      label: 'Semantic Similarity', 
      icon: Target,
      description: 'Find related content based on meaning and context'
    },
    { 
      id: 'categorical', 
      label: 'Smart Categorization', 
      icon: Hash,
      description: 'Automatic grouping and pattern recognition'
    }
  ]

  const getStatusIcon = () => {
    if (isAnalyzing) return <RefreshCw className="h-5 w-5 animate-spin text-blue-600" />
    if (similarityResults) return <CheckCircle className="h-5 w-5 text-green-600" />
    return <Clock className="h-5 w-5 text-gray-400" />
  }

  const getStatusText = () => {
    if (isAnalyzing) return 'Analyzing similarity patterns...'
    if (similarityResults) return 'Analysis complete'
    return 'Ready to analyze'
  }

  if (!datasetLoaded || !currentChatId) {
    return (
      <Card className="border-0 shadow-lg bg-white">
        <CardContent className="p-8 text-center">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <Brain className="h-8 w-8 text-gray-400" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Similarity Analysis
          </h3>
          <p className="text-gray-600 mb-4">
            Upload a dataset to this chat session to start similarity analysis
          </p>
          <Badge variant="secondary" className="bg-orange-100 text-orange-800">
            No dataset loaded
          </Badge>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Analysis Configuration */}
      <Card className="border-0 shadow-lg bg-white">
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
              <Brain className="h-5 w-5 text-purple-600" />
            </div>
            <div>
              <CardTitle className="text-xl font-semibold text-gray-900">
                Smart Similarity Detection
              </CardTitle>
              <p className="text-sm text-gray-600 mt-1">
                Advanced pattern recognition and duplicate detection for any dataset
              </p>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Analysis Type Selection */}
          <div>
            <label className="text-sm font-medium text-gray-700 mb-3 block">
              Analysis Type
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {analysisTypes.map((type) => {
                const Icon = type.icon
                return (
                  <div
                    key={type.id}
                    className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                      analysisType === type.id
                        ? 'border-purple-300 bg-purple-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setAnalysisType(type.id)}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                        analysisType === type.id ? 'bg-purple-200' : 'bg-gray-100'
                      }`}>
                        <Icon className={`h-4 w-4 ${
                          analysisType === type.id ? 'text-purple-600' : 'text-gray-600'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <h4 className="font-medium text-gray-900 text-sm">
                          {type.label}
                        </h4>
                        <p className="text-xs text-gray-600 mt-1">
                          {type.description}
                        </p>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Search Query */}
          <div>
            <label className="text-sm font-medium text-gray-700 mb-2 block">
              Focus Area (Optional)
            </label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Enter specific topic, keywords, or criteria to focus the analysis..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Leave empty to analyze the entire dataset
            </p>
          </div>

          {/* Similarity Threshold */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium text-gray-700">
                Similarity Threshold
              </label>
              <Badge variant="secondary" className="text-xs">
                {Math.round(threshold * 100)}%
              </Badge>
            </div>
            <input
              type="range"
              min="0.5"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>More matches (50%)</span>
              <span>Exact matches (95%)</span>
            </div>
          </div>

          {/* Status and Actions */}
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-3">
              {getStatusIcon()}
              <div>
                <p className="text-sm font-medium text-gray-900">
                  {getStatusText()}
                </p>
                {isAnalyzing && (
                  <div className="w-48 mt-2">
                    <Progress value={progress} className="h-2" />
                  </div>
                )}
              </div>
            </div>

            <Button
              onClick={analyzeSimilarity}
              disabled={isAnalyzing}
              className="bg-purple-600 hover:bg-purple-700"
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Zap className="h-4 w-4 mr-2" />
                  Start Analysis
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Results Display */}
      {similarityResults && (
        <Card className="border-0 shadow-lg bg-white">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                  <CheckCircle className="h-4 w-4 text-green-600" />
                </div>
                <div>
                  <CardTitle className="text-lg font-semibold text-gray-900">
                    Similarity Analysis Results
                  </CardTitle>
                  <p className="text-sm text-gray-600">
                    Found {similarityResults.total_matches || 0} similarity patterns
                  </p>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Badge variant="secondary" className="bg-green-100 text-green-800">
                  {analysisTypes.find(t => t.id === analysisType)?.label}
                </Badge>
                <Button size="sm" variant="outline" onClick={analyzeSimilarity}>
                  <RefreshCw className="h-3 w-3 mr-2" />
                  Re-analyze
                </Button>
              </div>
            </div>
          </CardHeader>

          <CardContent>
            {/* Analysis Summary */}
            {similarityResults.summary && (
              <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border border-blue-200">
                <h4 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
                  <TrendingUp className="h-4 w-4 text-blue-600" />
                  Analysis Summary
                </h4>
                <div className="text-sm text-gray-700 whitespace-pre-wrap">
                  {similarityResults.summary}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}