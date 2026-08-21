---
description: Build React dashboards with charts, auth, and real-time data
triggers: react, dashboard, frontend, web app, nextjs
---

# React Dashboard Skill

## Stack
- Next.js 14+ (App Router)
- Tailwind CSS + shadcn/ui
- Recharts or Chart.js for visualizations
- React Query for data fetching
- Zustand for state management

## Template: Dashboard Page
```tsx
"use client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";

const data = [
  { name: "Jan", value: 400 },
  { name: "Feb", value: 300 },
  { name: "Mar", value: 500 },
];

export default function Dashboard() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader><CardTitle>Total Users</CardTitle></CardHeader>
        <CardContent><p className="text-2xl font-bold">1,234</p></CardContent>
      </Card>
      <Card className="col-span-2">
        <CardHeader><CardTitle>Revenue</CardTitle></CardHeader>
        <CardContent>
          <BarChart data={data} width={500} height={300}>
            <XAxis dataKey="name" /><YAxis /><Tooltip />
            <Bar dataKey="value" fill="#8884d8" />
          </BarChart>
        </CardContent>
      </Card>
    </div>
  );
}
```

## Key Patterns
- Server Components by default, `"use client"` only when needed
- API routes in `app/api/` for backend logic
- Environment variables: `NEXT_PUBLIC_` for client, plain for server
- Deploy: Vercel (zero config) or Docker
