const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'

class APIService {
  async uploadFileForChat(file, chatId) {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('chat_id', chatId)

    const response = await fetch(`${API_BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.json()
  }

  async getChatStatus(chatId) {
    const response = await fetch(`${API_BASE_URL}/chat/${chatId}/status`, {
      method: 'GET',
      headers: {
        'Cache-Control': 'no-cache'
      }
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.json()
  }

  async getStatus() {
    const response = await fetch(`${API_BASE_URL}/status`, {
      method: 'GET',
      headers: {
        'Cache-Control': 'no-cache'
      }
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.json()
  }

  async createNewChat() {
    const response = await fetch(`${API_BASE_URL}/chat/new`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.json()
  }

  async getChatHistory() {
    const response = await fetch(`${API_BASE_URL}/chat/history`, {
      method: 'GET',
      headers: {
        'Cache-Control': 'no-cache'
      }
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.json()
  }

  async getChat(chatId) {
    const response = await fetch(`${API_BASE_URL}/chat/${chatId}`, {
      method: 'GET',
      headers: {
        'Cache-Control': 'no-cache'
      }
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.json()
  }

 async deleteChat(chatId) {
   const response = await fetch(`${API_BASE_URL}/chat/${chatId}/delete`, {
     method: 'DELETE',
     headers: {
       'Content-Type': 'application/json'
     }
   })

   if (!response.ok) {
     throw new Error(`HTTP error! status: ${response.status}`)
   }

   return await response.json()
 }

 async renameChat(chatId, newTitle) {
   const response = await fetch(`${API_BASE_URL}/chat/${chatId}/rename`, {
     method: 'PUT',
     headers: {
       'Content-Type': 'application/json'
     },
     body: JSON.stringify({ title: newTitle })
   })

   if (!response.ok) {
     throw new Error(`HTTP error! status: ${response.status}`)
   }

   return await response.json()
 }

 async getChatDatasetStatus(chatId) {
   const response = await fetch(`${API_BASE_URL}/chat/${chatId}/dataset-status`, {
     method: 'GET',
     headers: {
       'Cache-Control': 'no-cache'
     }
   })

   if (!response.ok) {
     throw new Error(`HTTP error! status: ${response.status}`)
   }

   return await response.json()
 }

 async sendQuery(question, chatId) {
   const response = await fetch(`${API_BASE_URL}/query`, {
     method: 'POST',
     headers: {
       'Content-Type': 'application/json'
     },
     body: JSON.stringify({
       question,
       chat_id: chatId
     })
   })

   if (!response.ok) {
     throw new Error(`HTTP error! status: ${response.status}`)
   }

   return await response.json()
 }

 // Add to apiService.js
async sendSimilarityQuery(question, chatId) {
    const response = await fetch(`${API_BASE_URL}/similarity-query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        question,
        chat_id: chatId
      })
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    return await response.json()
  }
}

export const apiService = new APIService()