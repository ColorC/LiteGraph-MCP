import type { ErrorInfo, ReactNode } from "react";
import { Button, Result, Typography } from "antd";
import { Component } from "react";

const { Paragraph, Text } = Typography;

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
    this.setState({ errorInfo });
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            padding: 40,
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            height: "100vh",
          }}
        >
          <Result
            status="error"
            title="UI发生崩溃 (Something went wrong)"
            subTitle="Application has encountered a critical error. Please refresh or contact support."
            extra={[
              <Button type="primary" key="reload" onClick={() => window.location.reload()}>
                刷新页面 (Reload)
              </Button>,
            ]}
          >
            <div
              style={{
                textAlign: "left",
                background: "#f5f5f5",
                padding: 20,
                borderRadius: 8,
                maxWidth: 800,
              }}
            >
              <Text type="danger" strong>
                Error: {this.state.error?.toString()}
              </Text>
              <Paragraph
                style={{
                  marginTop: 10,
                  fontSize: 12,
                  fontFamily: "monospace",
                  whiteSpace: "pre-wrap",
                }}
              >
                {this.state.errorInfo?.componentStack}
              </Paragraph>
            </div>
          </Result>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
