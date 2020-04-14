import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class LogService {

  constructor(private http: HttpClient) { }

  welcome = 'Test';

  getLogs() {
    return this.http.get('http://127.0.0.1:5000/logs/');
  }
}
