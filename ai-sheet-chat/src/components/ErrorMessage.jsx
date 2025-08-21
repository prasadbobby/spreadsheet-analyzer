import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { AlertTriangle, RefreshCw, Settings, Clock } from 'lucide-react'

export default function ErrorMessage({ error, onRetry }) {
  const getErrorIcon = (errorType) => {
    switch (errorType) {
      case 'RATE_LIMIT_EXCEEDED':
        return <Clock className="h-6 w-6 text-orange-600" />
      case 'GEMINI_API_ERROR':
        return <AlertTriangle className="h-6 w-6 text-red-600" />
      case 'GEMINI_API_TIMEOUT':
        return <Clock className="h-6 w-6 text-yellow-600" />
      case 'GEMINI_API_CONNECTION':
        return <AlertTriangle className="h-6 w-6 text-blue-600" />
      default:
        return <AlertTriangle className="h-6 w-6 text-gray-600" />
    }
  }

  const getErrorColor = (errorType) => {
    switch (errorType) {
      case 'RATE_LIMIT_EXCEEDED':
        return 'border-orange-200 bg-orange-50'
      case 'GEMINI_API_ERROR':
        return 'border-red-200 bg-red-50'
      case 'GEMINI_API_TIMEOUT':
        return 'border-yellow-200 bg-yellow-50'
      case 'GEMINI_API_CONNECTION':
        return 'border-blue-200 bg-blue-50'
      default:
        return 'border-gray-200 bg-gray-50'
    }
  }

  return (
    <Card className={`border-2 ${getErrorColor(error.error_type)}`}>
      <CardContent className="p-6">
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0">
            {getErrorIcon(error.error_type)}
          </div>
          <div className="flex-1">
            <div className="prose prose-sm max-w-none" 
                 dangerouslySetInnerHTML={{ __html: error.content }} />
            
            {onRetry && (
              <div className="mt-4 flex gap-2">
                <Button size="sm" onClick={onRetry} variant="outline">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Try Again
                </Button>
                {error.error_type === 'RATE_LIMIT_EXCEEDED' && (
                  <Button size="sm" variant="outline">
                    <Settings className="h-4 w-4 mr-2" />
                    API Settings
                  </Button>
                )}
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}