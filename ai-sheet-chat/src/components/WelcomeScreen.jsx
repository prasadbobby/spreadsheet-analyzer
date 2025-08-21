import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { 
  Brain, 
  Check, 
  Upload, 
  MessageSquare,
  Sparkles,
  TrendingUp,
  Database,
  BarChart3,
  Zap,
  GitMerge,
  Search,
  Users,
  DollarSign,
  Bug
} from 'lucide-react'

export default function WelcomeScreen() {
  const features = [
    { icon: MessageSquare, text: 'Natural language data queries' },
    { icon: Brain, text: 'Conversational AI analysis' },
    { icon: TrendingUp, text: 'Automatic pattern recognition' },
    { icon: BarChart3, text: 'Dynamic visualizations' },
    { icon: Database, text: 'Handle datasets up to 10GB' },
    { icon: Zap, text: 'Real-time insights generation' }
  ]

  const steps = [
    { icon: Upload, title: 'Upload Data', description: 'Drop your CSV or Excel file' },
    { icon: MessageSquare, title: 'Ask Questions', description: 'Use natural language to query' },
    { icon: Sparkles, title: 'Get Insights', description: 'Receive AI-powered analysis' }
  ]

  const datasetExamples = [
    {
      icon: Bug,
      title: 'JIRA Tickets',
      description: 'Bug tracking and project management',
      examples: [
        '"Find similar bugs to Connection refused error"',
        '"Show all critical tickets this month"',
        '"Analyze ticket trends over last quarter"'
      ]
    },
    {
      icon: DollarSign,
      title: 'Sales Data',
      description: 'Revenue and customer analysis',
      examples: [
        '"Show top performing products this quarter"',
        '"Find customers with highest revenue"',
        '"Analyze sales trends by region"'
      ]
    },
    {
      icon: Users,
      title: 'HR Records',
      description: 'Employee and workforce analytics',
      examples: [
        '"Show salary distribution by department"',
        '"Find employees with specific skills"',
        '"Analyze performance ratings trends"'
      ]
    },
    {
      icon: Database,
      title: 'Any Dataset',
      description: 'General purpose data analysis',
      examples: [
        '"Summarize the key patterns in this data"',
        '"Find similar or duplicate records"',
        '"Show me interesting trends and insights"'
      ]
    }
  ]

  return (
    <div className="flex items-center justify-center min-h-full p-8">
      <div className="max-w-6xl w-full space-y-8">
        {/* Hero Section */}
        <div className="text-center space-y-6">
          <div className="w-20 h-20 gradient-bg rounded-full flex items-center justify-center mx-auto shadow-lg">
            <Brain className="h-10 w-10 text-white" />
          </div>
          
          <div className="space-y-3">
            <h1 className="text-4xl font-bold text-gray-900">
              Welcome to <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-blue-600">AI Sheet Chat</span>
            </h1>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
              Your intelligent data analysis companion. Upload any dataset and have natural conversations about your data. 
            </p>
          </div>
        </div>

        {/* How it Works */}
        <Card className="border-0 shadow-lg bg-gradient-to-br from-blue-50 to-purple-50">
          <CardContent className="p-8">
            <h2 className="text-2xl font-bold text-center text-gray-900 mb-8">How It Works</h2>
            <div className="grid md:grid-cols-3 gap-6">
              {steps.map((step, index) => (
                <div key={index} className="text-center space-y-4">
                  <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto shadow-md">
                    <step.icon className="h-8 w-8 text-purple-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{step.title}</h3>
                    <p className="text-gray-600">{step.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Dataset Examples */}
        <div className="space-y-6">
          <h2 className="text-2xl font-bold text-center text-gray-900">Perfect for Any Dataset</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {datasetExamples.map((example, index) => (
              <Card key={index} className="border-0 shadow-lg hover:shadow-xl transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-purple-100 to-blue-100 rounded-lg flex items-center justify-center">
                      <example.icon className="h-5 w-5 text-purple-600" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900">{example.title}</h3>
                      <p className="text-xs text-gray-500">{example.description}</p>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-gray-700 mb-3">Example questions:</p>
                    {example.examples.map((question, qIndex) => (
                      <div key={qIndex} className="p-2 bg-gray-50 rounded text-xs text-gray-600 border-l-2 border-purple-200">
                        {question}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="border-0 shadow-lg">
            <CardContent className="p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Features</h3>
              <div className="space-y-3">
                {features.map((feature, index) => (
                  <div key={index} className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0">
                      <feature.icon className="h-4 w-4 text-green-600" />
                    </div>
                    <span className="text-gray-700">{feature.text}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-lg bg-gradient-to-br from-purple-50 to-pink-50">
            <CardContent className="p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Smart Analysis</h3>
              <div className="space-y-3">
                <div className="p-3 bg-white/70 rounded-lg border border-white/50">
                  <div className="flex items-center gap-2 mb-1">
                    <GitMerge className="h-4 w-4 text-purple-600" />
                    <span className="text-sm font-medium text-gray-800">Similarity Detection</span>
                  </div>
                  <p className="text-xs text-gray-600">Find duplicates and similar records automatically</p>
                </div>
                <div className="p-3 bg-white/70 rounded-lg border border-white/50">
                  <div className="flex items-center gap-2 mb-1">
                    <TrendingUp className="h-4 w-4 text-purple-600" />
                    <span className="text-sm font-medium text-gray-800">Trend Analysis</span>
                  </div>
                  <p className="text-xs text-gray-600">Automatic pattern recognition and insights</p>
                </div>
                <div className="p-3 bg-white/70 rounded-lg border border-white/50">
                  <div className="flex items-center gap-2 mb-1">
                    <Sparkles className="h-4 w-4 text-purple-600" />
                    <span className="text-sm font-medium text-gray-800">Conversational AI</span>
                  </div>
                  <p className="text-xs text-gray-600">Natural language understanding of your data</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* CTA */}
        <div className="text-center">
          <p className="text-gray-600 mb-4">Ready to start? Upload your data file to begin intelligent analysis.</p>
          <div className="flex items-center justify-center gap-2 text-purple-600">
            <Upload className="h-5 w-5" />
            <span className="font-medium">Drag & drop your file or click the upload area</span>
          </div>
          
        </div>
      </div>
    </div>
  )
}