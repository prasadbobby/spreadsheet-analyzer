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
  Zap
} from 'lucide-react'

export default function WelcomeScreen() {
  const features = [
    { icon: MessageSquare, text: 'Natural language data queries' },
    { icon: Brain, text: 'AI-powered intelligent analysis' },
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

  return (
    <div className="flex items-center justify-center min-h-full p-8">
      <div className="max-w-4xl w-full space-y-8">
        {/* Hero Section */}
        <div className="text-center space-y-6">
          <div className="w-20 h-20 gradient-bg rounded-full flex items-center justify-center mx-auto shadow-lg">
            <Brain className="h-10 w-10 text-white" />
          </div>
          
          <div className="space-y-3">
            <h1 className="text-4xl font-bold text-gray-900">
              Welcome to <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-blue-600">AI Sheet Chat</span>
            </h1>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto leading-relaxed">
              Transform your data into insights with conversational AI. Upload your spreadsheet and start asking questions in plain English.
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

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 gap-6">
          <Card className="border-0 shadow-lg">
            <CardContent className="p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Powerful Features</h3>
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
              <h3 className="text-xl font-bold text-gray-900 mb-4">Example Questions</h3>
              <div className="space-y-3">
                <div className="p-3 bg-white/70 rounded-lg border border-white/50">
                  <p className="text-sm text-gray-700">"Show me all employees with salary above $80,000"</p>
                </div>
                <div className="p-3 bg-white/70 rounded-lg border border-white/50">
                  <p className="text-sm text-gray-700">"What's the average performance rating by department?"</p>
                </div>
                <div className="p-3 bg-white/70 rounded-lg border border-white/50">
                  <p className="text-sm text-gray-700">"Find any unusual patterns in the data"</p>
                </div>
                <div className="p-3 bg-white/70 rounded-lg border border-white/50">
                  <p className="text-sm text-gray-700">"Create a summary of key insights"</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* CTA */}
        <div className="text-center">
          <p className="text-gray-600 mb-4">Ready to start? Upload your data file to begin the analysis.</p>
          <div className="flex items-center justify-center gap-2 text-purple-600">
            <Upload className="h-5 w-5" />
            <span className="font-medium">Drag & drop your file or click the upload area</span>
          </div>
        </div>
      </div>
    </div>
  )
}